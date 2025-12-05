# src/llm_api.py
import os
from groq import Groq

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY not found. Please set it as a secret in your Space settings."
    )

client = Groq(api_key=api_key)

DEFAULT_MODEL = "mixtral-8x7b-32768"  


def generate_llm_response(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> str:
    """
    Chat completion wrapper using Groq API (FAST + FREE).
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()
