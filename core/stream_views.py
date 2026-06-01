import collections
import inspect
import json
import uuid

import bleach
from django.http import StreamingHttpResponse, HttpResponse
from django.shortcuts import render
from django.template.loader import render_to_string
from django.urls import path
from django.conf import settings
from langchain_core.messages import HumanMessage, AIMessage

from django.urls import reverse_lazy
from langgraph.errors import InvalidUpdateError

from markdown import markdown  # pip install markdown

import logging

from core.agent.tools.agent import get_agent
from core.models import ChatThread

logger = logging.getLogger(f"ollama.{__name__}")


def check_model(agent_changed=False):
    html = render_to_string('core/steam.html#ai_model',
                            context={"model": settings.OLLAMA_MODEL, "agent_changed": agent_changed})
    logger.info(html)
    yield f"data: {html.replace("\n", "")}\n\n"


# You can customize allowed tags/attributes as needed
ALLOWED_TAGS = set(bleach.sanitizer.ALLOWED_TAGS) | {
    'p', 'pre', 'code', 'span', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'br', 'hr', 'img'
}
ALLOWED_ATTRIBUTES = {
    **bleach.sanitizer.ALLOWED_ATTRIBUTES,
    'a': ['href', 'title', 'rel', 'target'],
    'img': ['src', 'alt', 'title', 'width', 'height'],
    'code': ['class'],
    'span': ['class'],
}

MAX_HISTORY_MESSAGES = 20
THINKING_TARGET = "div_id_thinking_stream"
CONTENT_TARGET = "div_id_content_stream"
HISTORY_TARGET = "div_id_history_stream"
MESSAGE_SPACER = "<p>=========================================================================================</p>"


def _clean_message(string_buffer: str):
    html = markdown(string_buffer, extensions=['extra', 'codehilite', 'sane_lists'])
    # Sanitize the HTML to avoid XSS
    clean_thinking_html = bleach.clean(
        html,
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRIBUTES,
        strip=True
    )
    # '\n' have to be removed if this is to be sent as an SSE event
    # which must have the format "data: [something to send]\n\n"
    # if there are any '\n\n' in the message it won't work.
    html = clean_thinking_html.replace('\n', '')
    return html


def _clean_response(string_buffer: str, target: str, scroll: bool = True):
    html = _clean_message(string_buffer)
    return f'<hx-partial hx-target="#{target}" hx-swap="beforeend{" scroll:bottom" if scroll else ""}">{html}</hx-partial>'


def get_history(request, agent):
    thread = get_thread()
    config = {"configurable": {"thread_id": thread.thread_id}}

    # Get the latest snapshot of the state
    state = agent.get_state(config)

    # Access the 'messages' list from the state values
    # Note: Ensure the key 'messages' matches what you defined in your State schema
    return state.values.get("messages", [])


def _process_chunk(stream, target, max_buffer_size=16_384):
    """
    Process an iterable `stream` of chunks and yield SSE 'data: ...\n\n' entries.
    This function:
    - coerces non-string chunks to str/json safely,
    - splits chunks by newline(s) and flushes appropriately,
    - guards against endlessly growing buffers by flushing if buffer grows too large.
    """
    chunk_buf = ""
    # If the stream is a single string (common), allow it to be processed as an iterable of one.
    if isinstance(stream, str):
        iterable = (stream,)
    elif isinstance(stream, collections.abc.Iterator) and not isinstance(stream, collections.abc.Sequence):
        iterable = stream
    else:
        # Accept lists, tuples, generators equally
        iterable = stream

    for chunk in iterable:
        try:
            if chunk is None:
                continue
            # Normalize chunk to string
            if not isinstance(chunk, str):
                try:
                    # For dict/list-like chunks, prefer compact JSON
                    if isinstance(chunk, (dict, list)):
                        chunk = json.dumps(chunk, separators=(',', ':'), ensure_ascii=False)
                    else:
                        chunk = str(chunk)
                except Exception:
                    chunk = str(chunk)

            # If chunk contains one or more newlines, flush every complete line
            if '\n' in chunk:
                parts = chunk.split('\n')
                # flush first line as continuation of buffer
                chunk_buf += parts[0]
                if chunk_buf:
                    yield f'data: {_clean_response(chunk_buf, target)}\n\n'
                # flush intermediate complete lines
                for mid in parts[1:-1]:
                    if mid:
                        yield f'data: {_clean_response(mid, target)}\n\n'
                # start new buffer with the last (possibly partial) segment
                chunk_buf = parts[-1]
            else:
                chunk_buf += chunk

            # guard: flush if buffer grows too large (avoids memory blowup)
            if len(chunk_buf) > max_buffer_size:
                yield f'data: {_clean_response(chunk_buf, target)}\n\n'
                chunk_buf = ""
        except Exception:
            logger.exception("Error processing chunk for target=%s", target)
            # continue to next chunk instead of stopping the whole stream
            continue

    # flush remainder
    if chunk_buf:
        try:
            yield f'data: {_clean_response(chunk_buf, target)}\n\n'
        except Exception:
            logger.exception("Error flushing final buffer for target=%s", target)


def _client_stream(stream):
    html = render_to_string('core/partials/spinner.html').replace('\n', '')

    yield f'data: <hx-partial hx-target="#{HISTORY_TARGET}_spinner" hx-swap="innerHTML">{html}</hx-partial>\n\n'
    yield f'data: <hx-partial hx-target="#{THINKING_TARGET}_spinner" hx-swap="innerHTML">{html}</hx-partial>\n\n'
    msg_update_url = reverse_lazy("core:post_message")
    yield f'data: <hx-partial hx-target="#{CONTENT_TARGET}_spinner" hx-swap="innerHTML"><span hx-get="{msg_update_url}" hx-trigger="load"></span>{html}</hx-partial>\n\n'

    logger.info("Streaming reasoning/messages, list tool calls")
    message_id = None
    is_thinking = False
    for name, item in stream.interleave("messages", "tool_calls"):
        try:
            if name == "messages":
                if item.message_id != message_id:
                    message_id = item.message_id
                    logger.info(f"message id updated {message_id}")

                if hasattr(item, 'reasoning'):
                    yield from _process_chunk(item.reasoning, THINKING_TARGET)
                if hasattr(item, 'text'):
                    yield from _process_chunk(item.text, CONTENT_TARGET)
            elif name == "tool_calls":
                if item.error:
                    logger.error("Tool %s failed: %s", item.tool_name, item.error)
                    yield f"data: {_clean_response(f'<p>Error in tool: {str(item.error).replace('\\n', '')}</p>', THINKING_TARGET)}\n\n"
                    continue

                # announce the tool call (protected)
                try:
                    yield f"data: {_clean_response(f'<p> &gt;&gt; Calling tool: <code>{item.tool_name}({item.input})</code></p>', THINKING_TARGET)}\n\n"
                except Exception:
                    logger.exception("Failed to format tool call announcement for %s", getattr(item, "tool_name", "<unknown>"))
                    # best-effort fallback announcement
                    tool_name_id = getattr(item, "tool_name", "<unknown>")
                    yield f"data: {_clean_response(f'<p> &gt;&gt; Calling tool: {tool_name_id}</p>', THINKING_TARGET)}\n\n"

                out_deltas = getattr(item, "output_deltas", None)
                logger.debug("Tool %s output_deltas: type=%s repr=%s", item.tool_name, type(out_deltas), repr(out_deltas)[:400])

                # nothing to stream
                if out_deltas is None:
                    continue

                # handle concrete sequences quickly
                if isinstance(out_deltas, (list, tuple)):
                    try:
                        yield from _process_chunk(out_deltas, THINKING_TARGET)
                    except Exception:
                        logger.exception("Error processing list output_deltas for tool %s", item.tool_name)
                        yield f"data: {_clean_response(f'<p>Tool {item.tool_name} produced output that could not be processed.</p>', THINKING_TARGET)}\n\n"
                    continue

                # detect async outputs
                if inspect.isasyncgen(out_deltas) or inspect.iscoroutine(out_deltas):
                    logger.warning("Tool %s returned async output; synchronous streaming not supported", item.tool_name)
                    yield f"data: {_clean_response(f'<p>Tool {item.tool_name} returned asynchronous output. Cannot stream it here.</p>', THINKING_TARGET)}\n\n"
                    continue

                # try to iterate safely (generators / iterables / single values)
                try:
                    iterator = iter(out_deltas)
                except TypeError:
                    # single non-iterable: process as a single chunk
                    yield from _process_chunk((out_deltas,), THINKING_TARGET)
                    continue

                # iterate generator/iterator with logging + max guard
                max_items = 10000
                count = 0
                try:
                    for delta in iterator:
                        count += 1
                        logger.debug("Tool %s delta #%d type=%s repr=%s", item.tool_name, count, type(delta), repr(delta)[:300])
                        # process each delta as its own chunk (keeps buffering predictable)
                        yield from _process_chunk((delta,), THINKING_TARGET)

                        # heartbeat to keep connection/proxies alive
                        if (count % 10) == 0:
                            yield f"data: {_clean_response('', THINKING_TARGET)}\n\n"

                        if count >= max_items:
                            logger.warning("Tool %s produced more than %d deltas; truncating", item.tool_name, max_items)
                            yield f"data: {_clean_response(f'<p>Tool {item.tool_name} produced too many messages; truncating.</p>', THINKING_TARGET)}\n\n"
                            break
                except Exception:
                    logger.exception("Exception while iterating output_deltas for tool %s", item.tool_name)
                    yield f"data: {_clean_response(f'<p>Tool {item.tool_name} streaming failed while iterating its output.</p>', THINKING_TARGET)}\n\n"
        except Exception as ex:
            logger.exception(ex)
            continue

    yield f'data: <hx-partial hx-target="#{CONTENT_TARGET}" hx-swap="beforeend">{MESSAGE_SPACER}</hx-partial>\n\n'
    yield f'data: <hx-partial hx-target="#{THINKING_TARGET}" hx-swap="beforeend">{MESSAGE_SPACER}</hx-partial>\n\n'

    yield f'data: <hx-partial hx-target="#{HISTORY_TARGET}_spinner" hx-swap="innerHTML scroll:bottom"></hx-partial>\n\n'
    yield f'data: <hx-partial hx-target="#{THINKING_TARGET}_spinner" hx-swap="innerHTML scroll:bottom"></hx-partial>\n\n'
    yield f'data: <hx-partial hx-target="#{CONTENT_TARGET}_spinner" hx-swap="innerHTML scroll:bottom"></hx-partial>\n\n'

def _process_content_message_history(request):
    unprocessed_messages = [msg.content for msg in get_history(request, get_agent()) if (type(msg) is HumanMessage or type(msg) is AIMessage)]
    messages = [{'html':msg if type(msg)==str else (_clean_message(msg[0]['text']) + MESSAGE_SPACER), 'role':'user' if type(msg)==str else 'ai'}
                for msg in unprocessed_messages if (type(msg)==str or (type(msg)==list and msg[0]['type']=='text'))]
    return messages

def index(request):
    model = settings.OLLAMA_MODEL
    thinking_messages = []
    his_msg = ""
    messages = _process_content_message_history(request)
    html = render(request, 'core/steam.html',
                  context={
                      "model": model,
                      "messages": messages,
                      "thinking": thinking_messages,
                      "history": his_msg
                  })
    return HttpResponse(html)


class ChatThreads:
    pass


def get_thread():
    if not (thread := ChatThread.objects.first()):
        thread = ChatThread.objects.create(thread_id=str(uuid.uuid4()))
    return thread

def start_stream(request):
    chat_message = request.POST.get('chat_message', "") or ""
    if not chat_message:
        return HttpResponse(status=400)

    msg = HumanMessage(content=chat_message)

    # Pass this config into stream_events
    thread = get_thread()
    config = {"configurable": {"thread_id": thread.thread_id}}

    stream = get_agent().stream_events({"messages": msg}, version='v3', config=config)
    return StreamingHttpResponse(_client_stream(stream), content_type="text/event-stream")


def clear_history(request):
    thread = get_thread()
    config = {"configurable": {"thread_id": thread.thread_id}}

    # Explicitly clear the state
    try:
        # Let the agent/library pick the appropriate node automatically.
        get_agent().update_state(config, {"messages": []}, as_node="agent")
        logger.info("Cleared agent messages for thread %s", thread.thread_id)
    except InvalidUpdateError as ex:
        # Log full exception for debugging
        logger.exception("Failed to clear agent history: %s", ex)

    html = f'<hx-partial hx-target="#{THINKING_TARGET}" hx-swap="innerHTML scroll:bottom"></hx-partial>'
    html += f'<hx-partial hx-target="#{CONTENT_TARGET}" hx-swap="innerHTML scroll:bottom"></hx-partial>'
    html += f'<hx-partial hx-target="#{HISTORY_TARGET}" hx-swap="innerHTML scroll:bottom"></hx-partial>'

    return HttpResponse(html)


def clear_agent_history(request):

    def chat_update():

        his_msg = ""
        yield f'data: <hx-partial hx-target="#{THINKING_TARGET}" hx-swap="innerHTML scroll:bottom"></hx-partial>\n\n'
        yield f"data: {_clean_response(his_msg, HISTORY_TARGET)}\n\n"

    return StreamingHttpResponse(chat_update(), content_type="text/event-stream")


def post_message(request):
    """ post the last user message to the Chat area """
    message = get_history(request, get_agent())
    if len(message) > 0:
        message = message[-1]

    if type(message) != HumanMessage:
        return HttpResponse()
    message = {"html": message.content, "role": "user"}
    html = render_to_string('core/steam.html#post_msg', context={"messages": [message]})
    return HttpResponse(html)

urlpatterns = [
    path('', index, name="stream_index"),
    path('stream/', index, name="stream_index"),
    path('stream/start/', start_stream, name="start_stream"),
    path('stream/clear/history/', clear_history, name="clear_history"),
    path('stream/clear/history/agent/', clear_agent_history, name="clear_agent_history"),
    path('stream/post/', post_message, name='post_message'),
]
