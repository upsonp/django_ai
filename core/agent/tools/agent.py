import re
from pathlib import Path
from typing import Callable, Any, Union, Optional, List

from django.conf import settings
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import StateT
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.runtime import Runtime
from langgraph.typing import ContextT

from core.agent.tools import directories

import logging

logger = logging.getLogger(f"ollama.{__name__}")
memory = MemorySaver()


class AgentErrorLogger(AgentMiddleware):
    def after_agent(
            self, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        return super().after_agent(state, runtime)

    def after_tool_error(self, error: Exception, tool_name: str):
        # This logs the error, but the agent still gets the error
        # in its history to continue the loop
        logging.error(f"Tool {tool_name} raised: {error}")


SYSTEM_BLOCK_RE = re.compile(r'SYSTEM\s*"""\s*(.*?)\s*"""', re.DOTALL | re.IGNORECASE)

def get_agent_tools() -> list[Callable]:
    return [
        directories.ls_directories, directories.ls_files, directories.ls_recursive_list, directories.read_text_file
    ]

def get_agent():
    base_model = settings.OLLAMA_CLIENT
    kwargs = {
        "model": base_model,
        "tools": get_agent_tools(),
        "checkpointer": memory,
        "middleware": [AgentErrorLogger()]
    }
    if settings.SINGLE_AGENT is None:
        settings.SINGLE_AGENT = create_agent(**kwargs)

    return settings.SINGLE_AGENT
