import collections
import inspect
import json
import time
import uuid

import bleach
from bs4 import BeautifulSoup
from django.http import StreamingHttpResponse, HttpResponse
from django.shortcuts import render
from django.template.loader import render_to_string
from django.urls import path
from django.conf import settings
from langchain_core.messages import HumanMessage, AIMessage

from django.urls import reverse_lazy
from langchain_core.stores import InMemoryStore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import InvalidUpdateError
from langgraph.types import Command

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
    'p', 'pre', 'code', 'span', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'br', 'hr', 'img',
    'table', 'thead', 'tbody', 'tfoot', 'tr', 'th', 'td', 'caption', 'col', 'colgroup'
}
ALLOWED_ATTRIBUTES = {
    **bleach.sanitizer.ALLOWED_ATTRIBUTES,
    'a': ['href', 'title', 'rel', 'target'],
    'img': ['src', 'alt', 'title', 'width', 'height'],
    'code': ['class'],
    'span': ['class'],
    # table-specific attributes that are commonly generated or useful
    'table': ['class', 'border', 'cellpadding', 'cellspacing', 'summary'],
    'th': ['colspan', 'rowspan', 'scope', 'class', 'align'],
    'td': ['colspan', 'rowspan', 'class', 'align'],
    'caption': ['class'],
    'col': ['span', 'class'],
    'colgroup': ['class'],
}

MESSAGE_SPACER = "<p>=========================================================================================</p>"


def _clean_message(string_buffer: str):
    html = markdown(string_buffer, extensions=['extra', 'codehilite', 'sane_lists', 'tables'])

    # Sanitize the HTML to avoid XSS
    clean_thinking_html = bleach.clean(
        html,
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRIBUTES,
        strip=True
    )

    # Post-process HTML to add CSS classes automatically
    soup = BeautifulSoup(clean_thinking_html, "html.parser")

    # Example: add bootstrap-like classes to all tables
    for table in soup.find_all("table"):
        classes = table.get("class", [])
        # ensure we don't duplicate classes
        for cls in ("table", "table-striped"):
            if cls not in classes:
                classes.append(cls)
        table['class'] = classes
    # '\n' have to be removed if this is to be sent as an SSE event
    # which must have the format "data: [something to send]\n\n"
    # if there are any '\n\n' in the message it won't work.
    html = clean_thinking_html.replace('\n', '')
    return html


def _stream_output(target, message, swap="beforeend scroll:bottom"):
    return f"<hx-partial hx-target=\"#div_id_{target}\" hx-swap=\"{swap}\">{message}</hx-partial>"

def _stream_interleave(stream):
    for name, item in stream.interleave("messages", "tool_calls"):
        try:
            if name == "messages":
                if hasattr(item, 'reasoning'):
                    think_buf = ""
                    tag = "thinking"
                    for chunk in item.reasoning:
                        think_buf += chunk if chunk else ""
                        yield f'data: {_stream_output(f"{tag}_stream", chunk.replace('\n', '<br>'))}\n\n'

                    if item.done:
                        yield f'data: {_stream_output(f"{tag}_content", _clean_message(think_buf))}\n\n'
                        yield f'data: {_stream_output(f"{tag}_content", MESSAGE_SPACER)}\n\n'
                        yield f'data: {_stream_output(f"{tag}_stream", "", "innerHTML")}\n\n'
                        thinking_buf = ""

                if hasattr(item, 'text'):
                    text_buf = ""
                    tag = "text"
                    for chunk in item.text:
                        text_buf += chunk if chunk else ""
                        yield f'data: {_stream_output(f"{tag}_stream", chunk.replace('\n', '<br>'))}\n\n'

                    if item.done:
                        yield f'data: {_stream_output(f"{tag}_content", _clean_message(text_buf))}\n\n'
                        yield f'data: {_stream_output(f"{tag}_content", MESSAGE_SPACER)}\n\n'
                        yield f'data: {_stream_output(f"{tag}_stream", "", "innerHTML")}\n\n'
                        text_buf = ""

            elif name == "tool_calls":
                if item.error:
                    logger.error("Tool %s failed: %s", item.tool_name, item.error)
                    yield f"data: {_stream_output("thinking_content", _clean_message(f'<p>Error in tool: {str(item.error).replace('\\n', '')}</p>'))}\n\n"
                    continue

                # announce the tool call (protected)
                try:
                    message = _clean_message(f'<p> &gt;&gt; Calling tool: <code>{item.tool_name}({item.input})</code></p>')
                    yield f"data: {_stream_output("thinking_content", message)}\n\n"
                except Exception:
                    logger.exception("Failed to format tool call announcement for %s", getattr(item, "tool_name", "<unknown>"))
                    # best-effort fallback announcement
                    tool_name_id = getattr(item, "tool_name", "<unknown>")
                    message = f'<p> &gt;&gt; Calling tool: {tool_name_id}</p>'
                    yield f"data: {_stream_output("thinking_content", message)}\n\n"
                    yield f"data: {_stream_output("text_content", message)}\n\n"

        except Exception as ex:
            logger.exception(ex)
            continue


def _flush_buffer(tag, buffer):
    yield f'data: {_stream_output(f"{tag}_content", _clean_message(buffer))}\n\n'
    yield f'data: {_stream_output(f"{tag}_content", MESSAGE_SPACER)}\n\n'
    yield f'data: {_stream_output(f"{tag}_stream", "", "innerHTML")}\n\n'

def _stream_raw(stream):
    text_buf = ""
    thinking_buf = ""

    iterator = iter(stream)
    while True:
        # Catch errors raised when advancing the iterator (these were
        # previously uncaught and can terminate the generator unexpectedly)
        try:
            event = next(iterator)
        except StopIteration:
            if thinking_buf:
                yield from _flush_buffer("thinking", thinking_buf)
                thinking_buf = ""
            if text_buf:
                yield from _flush_buffer("text", text_buf)
                text_buf = ""
            logger.info("End of stream")
            break
        except Exception as e:
            # Iterator raised an unexpected exception (connection/library/agent error)
            logger.exception("Stream iterator raised an exception; stopping stream")
            # Notify client of the error via SSE (best-effort)
            try:
                err_html = _clean_message(f'<p><strong>Stream error:</strong> {str(e)}</p>')
            except Exception:
                err_html = '<p><strong>Stream error</strong></p>'
            yield f'data: {_stream_output("text_content", err_html)}\n\n'

            # Flush buffered partial content so client sees what was built so far
            if thinking_buf:
                try:
                    yield from _flush_buffer("thinking", thinking_buf)
                except Exception:
                    logger.exception("Failed to flush thinking buffer after iterator error")
                thinking_buf = ""
            if text_buf:
                try:
                    yield from _flush_buffer("text", text_buf)
                except Exception:
                    logger.exception("Failed to flush text buffer after iterator error")
                text_buf = ""
            break

        # Process each event, but still protect the processing itself
        try:
            if not isinstance(event, dict):
                logger.debug("Skipping non-dict event: %r", event)
                continue

            method = event.get("method")
            if method == "tools":
                params = event.get("params", {})
                data = params.get("data", {})
                if data.get("event") == "tool-started":
                    tool_name = data.get("tool_name")
                    tool_args = data.get("input")
                    message = _clean_message(
                        f'<p> &gt;&gt; Calling tool: <code>{tool_name}({str(tool_args)})</code></p>')
                    yield f"data: {_stream_output('thinking_content', message)}\n\n"
                    yield f'data: {_stream_output("thinking_content", MESSAGE_SPACER)}\n\n'
                    yield f"data: {_stream_output('text_content', message)}\n\n"
                    yield f'data: {_stream_output("text_content", MESSAGE_SPACER)}\n\n'
                continue

            if method != "messages":
                logger.info("Event method: %s", method)
                continue

            data_list = event.get("params", {}).get("data", [])
            if data_list == []:
                # diagnostic: give the producer a short chance to supply data
                logger.debug("Received empty data_list; sleeping briefly before continue")
                time.sleep(0.05)  # 50ms; tune or remove after debugging
                # attempt to fetch again if event supports it (best-effort)
                # but most streams will provide proper events — we keep a simple continue here
                continue

            data = data_list[0]
            if not isinstance(data, dict):
                continue

            event_type = data.get("event")
            if event_type == "content-block-delta":
                block = data.get("delta") or {}

                if block.get("type") == "reasoning-delta":
                    chunk = block.get("reasoning", "") or ""
                    thinking_buf += chunk
                    yield f'data: {_stream_output(f"thinking_stream", chunk.replace(chr(10), "<br>"))}\n\n'

                elif block.get("type") == "text-delta":
                    chunk = block.get("text", "") or ""
                    text_buf += chunk
                    yield f'data: {_stream_output(f"text_stream", chunk.replace(chr(10), "<br>"))}\n\n'
                continue
            elif event_type == "message-finish":
                logger.info("Block type: %s", event_type)
                if thinking_buf:
                    yield from _flush_buffer("thinking", thinking_buf)
                    thinking_buf = ""
                if text_buf:
                    yield from _flush_buffer("text", text_buf)
                    text_buf = ""

                continue
            else:
                logger.info("Block type: %s", event_type)

        except Exception:
            logger.exception("Error processing stream event: %r", event)
            # continue to next event
            continue

def _client_stream(stream):
    html = render_to_string('core/partials/spinner.html').replace('\n', '')

    yield f'data: <hx-partial hx-target="#div_id_history_spinner" hx-swap="innerHTML">{html}</hx-partial>\n\n'
    yield f'data: <hx-partial hx-target="#div_id_thinking_spinner" hx-swap="innerHTML">{html}</hx-partial>\n\n'
    msg_update_url = reverse_lazy("core:post_message")
    yield f'data: <hx-partial hx-target="#div_id_text_spinner" hx-swap="innerHTML"><span hx-get="{msg_update_url}" hx-trigger="load"></span>{html}</hx-partial>\n\n'

    logger.info("Streaming reasoning/messages, list tool calls")
    yield from _stream_raw(stream)

    yield f'data: <hx-partial hx-target="#div_id_history_spinner" hx-swap="innerHTML"></hx-partial>\n\n'
    yield f'data: <hx-partial hx-target="#div_id_thinking_spinner" hx-swap="innerHTML"></hx-partial>\n\n'
    yield f'data: <hx-partial hx-target="#div_id_text_spinner" hx-swap="innerHTML"></hx-partial>\n\n'

def _process_content_message_history(request):
    unprocessed_messages = [msg.content for msg in get_history() if (type(msg) is HumanMessage or type(msg) is AIMessage)]
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
    if (thread := ChatThread.objects.last()) is None:
        thread = ChatThread.objects.create(thread_id=str(uuid.uuid4()))

    return thread

def get_config() -> dict:
    thread = get_thread()
    return {"configurable": {"thread_id": thread.thread_id}}

def start_stream(request):
    chat_message = request.POST.get('chat_message', "") or ""
    if not chat_message:
        return HttpResponse(status=400)

    msg = HumanMessage(content=chat_message)

    # Pass this config into stream_events
    config = get_config()

    stream = get_agent().stream_events({"messages": msg}, version='v3', config=config)
    return StreamingHttpResponse(_client_stream(stream), content_type="text/event-stream")


def get_history():
    config = get_config()

    # Get the latest snapshot of the state
    state = get_agent().get_state(config)

    # Access the 'messages' list from the state values
    # Note: Ensure the key 'messages' matches what you defined in your State schema
    return state.values.get("messages", [])


def clear_history(request):
    # Explicitly clear the state
    try:
        thread = ChatThread.objects.first()
        thread.thread_id = str(uuid.uuid4())
        thread.save()

        agent = get_agent()
        config = get_config()
        # Let the agent/library pick the appropriate node automatically.

        agent.update_state(config, Command(update={"messages": []}), as_node="__start__")
        logger.info("Cleared agent messages for thread %s", str(config))
    except InvalidUpdateError as ex:
        # Log full exception for debugging
        logger.exception("Failed to clear agent history: %s", ex)

    html = f'<hx-partial hx-target="#div_id_thinking_content" hx-swap="innerHTML scroll:bottom"></hx-partial>'
    html += f'<hx-partial hx-target="#div_id_text_content" hx-swap="innerHTML scroll:bottom"></hx-partial>'
    html += f'<hx-partial hx-target="#div_id_histroy_content" hx-swap="innerHTML scroll:bottom"></hx-partial>'

    return HttpResponse(html)


def clear_agent_history(request):
    return HttpResponse("Not implemented")


def post_message(request):
    """ post the last user message to the Chat area """
    message = get_history()
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
