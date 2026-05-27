import json

import ollama
import bleach

from django.conf import settings
from config.settings import ollama_client as ollama

from django.http import HttpResponse
from django.shortcuts import render
from django.urls import path

# top of file imports (add)
from markdown import markdown as md_to_html
from django.utils.safestring import mark_safe

import logging

from core.agent.tools.directories import ls_directories, ls_files
from core.agent.tools.weather import get_temperature

logger = logging.getLogger(f"ollama.{__name__}")

MAX_HISTORY_MESSAGES = 20

# helper: convert markdown to sanitized HTML
def markdown_to_safe_html(text: str) -> str:
    """
    Convert markdown text to HTML and sanitize with bleach.
    Returns a Django-safe string (mark_safe).
    """
    if not text:
        return ""

    # Convert Markdown -> HTML (enable fenced code, nl2br)
    raw_html = md_to_html(text, extensions=["fenced_code", "codehilite", "nl2br"])

    # Allow list (extend as needed). Keep this minimal for safety.
    allowed_tags = list(bleach.sanitizer.ALLOWED_TAGS) + [
        "p", "pre", "code", "br", "hr",
        "h1", "h2", "h3", "h4", "h5", "h6",
        "img", "table", "thead", "tbody", "tr", "th", "td",
        "blockquote", "ul", "ol", "li", "strong", "em"
    ]
    # Allowed attributes for tags
    allowed_attrs = {
        "a": ["href", "title", "rel", "target"],
        "img": ["src", "alt", "title", "width", "height"],
        "th": ["colspan", "rowspan"],
        "td": ["colspan", "rowspan"],
    }

    # Clean the HTML, remove disallowed tags/attrs, allow only safe protocols
    cleaned = bleach.clean(
        raw_html,
        tags=allowed_tags,
        attributes=allowed_attrs,
        protocols=["http", "https", "mailto"],
        strip=True,
    )

    # Convert plaintext links to anchors
    linkified = bleach.linkify(cleaned)

    # Mark safe for Django templates (we already sanitized)
    return mark_safe(linkified)

def _get_history(request):
    """Return message history list stored in session; ensure system prompt present once."""
    history = request.session.get('chat_history', [])

    # Ensure the system prompt is present at the start once
    return history

def _save_history(request, history):
    """Trim and save history back to session."""
    # Keep only the last MAX_HISTORY_MESSAGES items
    if len(history) > MAX_HISTORY_MESSAGES:
        history = history[-MAX_HISTORY_MESSAGES:]
        # Make sure the system prompt is still at index 0

    request.session['chat_history'] = history
    # Optionally set session.modified = True
    request.session.modified = True
    return history



def chat_page(request):
    return HttpResponse(render(request, "core/agent_chat.html"))


tool_list = [get_temperature, ls_directories, ls_files]
tool_dict = {tool.__name__: {"tool": tool} for tool in tool_list}

tool_dict["get_temperature"]["serializer"] = lambda obj: str(obj)
tool_dict["ls_directories"]["serializer"] = lambda obj: json.dumps({"directories": obj})
tool_dict["ls_files"]["serializer"] = lambda obj: json.dumps({"files": obj})

def tool_call(response, messages):
    message_content = ""
    message_thought = ""
    for call in response.message.tool_calls:
        logger.info(f"A.I is requesting a tool call {call.function.name}")

        result = tool_dict[call.function.name]["tool"](**call.function.arguments)
        serialize = tool_dict[call.function.name]["serializer"](result)

        # add the tool result to the messages
        messages.append({"role": "assistant", "content": serialize})
        messages.append({"role": "tool", "name": call.function.name, "content": serialize})

        final_response = ollama.chat(model=settings.OLLAMA_MODEL, messages=messages, tools=tool_list,
                                     think=settings.ENABLE_THINKING, stream=True)

        final_response_content = final_response.message.get('content', "")
        messages.append({"role": "assistant", "content": final_response_content})

        message_content += final_response_content or "" + "\n\n"
        message_thought += final_response.message.get('thinking', "") or "" + "\n\n"
        #
        # if final_response.message.tool_calls:
        #     sub_message_content, sub_message_thought = tool_call(response, messages)
        #     message_content += sub_message_content
        #     message_thought += sub_message_thought

    return message_content, message_thought


def agent_chat(request):
    chat_message = request.POST.get('chat_message', "") or ""
    logger.info(f"Chat message received: {chat_message}")

    # Only accept POST from the form
    if request.method != 'POST':
        return HttpResponse(status=405)

    if not chat_message:
        return HttpResponse(status=400)

    # Load history from session, ensures system prompt present
    history = _get_history(request)
    history.append({'role': 'user', 'content': chat_message})

    if len(history) > MAX_HISTORY_MESSAGES:
        history = history[-MAX_HISTORY_MESSAGES:]

    response = ollama.chat(model=settings.OLLAMA_MODEL, messages=history, tools=tool_list,
                           think=settings.ENABLE_THINKING, stream=True)

    message_content = response.message.get('content', "") or ""
    message_thought = response.message.get('thinking', "") or ""

    if response.message.tool_calls:
        # only recommended for models which only return a single tool call
        tool_content, tool_thought = tool_call(response, history)
        message_content += tool_content + "\n\n"
        message_thought += tool_thought + "\n\n"

    _save_history(request, history)

    if not message_content and not message_thought:
        logger.info("Chat message content empty")
        return HttpResponse()

    logger.info(f"Chat message content: {message_content}")
    logger.info(f"Chat message thought: {message_thought}")

    # Convert to safe HTML
    message_content_html = markdown_to_safe_html(message_content)
    message_thought_html = markdown_to_safe_html(message_thought)

    context = {"chat_message": message_content_html, "chat_thought": message_thought_html}
    html = render(request, 'core/partials/agent_chat_container.html#chat-items', context=context)

    return HttpResponse(html)


def clear_chat(request):
    history = []
    _save_history(request, history)
    return HttpResponse(render(request, "core/partials/agent_chat_container.html"))

urlpatterns = [
    path('chat/', chat_page, name='chat'),
    path('chat/submit/mesage/', agent_chat, name='agent_chat'),
    path('chat/clear/', clear_chat, name='clear_chat'),
]
