import requests
from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL, ENABLE_THINKING

def load_system_prompt():
    with open("core/agent/prompts/todo_prompt.md", "r", encoding="utf-8") as file:
        return file.read()

def make_request(title: str, description: str) -> list[str]:
    system_prompt = load_system_prompt()

    user_prompt = f"""
    Generate a comma separated list of todo items based on the following title and description.\n\n
    <example_title>
    { title }
    </example_title>
    <example_description>
    { description }
    </example_description>
    
    return only a comma separated list of items
    """

    if not ENABLE_THINKING:
        user_prompt = f"/no_think {user_prompt}"

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "think": ENABLE_THINKING,
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 0.1,
            "num_ctx": 1024,
        },
    }

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=180,
    )

    response.raise_for_status()

    raw_response = response.json()["message"]["content"]
    return raw_response.split(", ")