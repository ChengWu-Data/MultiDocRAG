"""
LLM API wrapper for the Multi-Document RAG system.

We use the Groq client to talk to open-weight models such as LLaMA 3.1.
"""

from __future__ import annotations

import os
from typing import Optional

from groq import Groq


# Human-readable names exposed in the Streamlit UI
AVAILABLE_MODELS = {
    "LLaMA 3.1 8B (fast)": "llama-3.1-8b-instant",
    "LLaMA 3.1 70B (better)": "llama-3.1-70b-versatile",
    "Mixtral 8x7B (long context)": "mixtral-8x7b-32768",
}

DEFAULT_MODEL = AVAILABLE_MODELS["LLaMA 3.1 8B (fast)"]

DEFAULT_SYSTEM_PROMPT = """
You are a careful teaching assistant for a data science / NLP course project.

- You must answer **only** based on the provided context from the documents.
- If the context is not enough to answer confidently, say:
  "I don't know based on the documents."
- Prefer short, precise answers (2–5 sentences) unless the question explicitly
  asks for a detailed explanation.
- When the user seems to ask about the implementation details of this app,
  you may answer at a high level but avoid inventing code that does not exist.
""".strip()


def _build_user_prompt(question: str, context: Optional[str]) -> str:
    if context:
        return (
            "You are given some context chunks retrieved from a PDF corpus.\n"
            "Use ONLY these chunks to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
    else:
        # Baseline mode: still remind the model to be cautious
        return (
            "Answer the user's question as best as you can.\n"
            "If you are not reasonably confident, say "
            "\"I am not sure\" instead of hallucinating details.\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )


def generate_llm_response(
    question: str,
    context: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 512,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """
    Call the Groq chat completion API and return the assistant's text reply.

    Parameters
    ----------
    question:
        The user question.
    context:
        Optional context string from the retriever (RAG mode). If None, the
        model runs in baseline mode.
    model_name:
        Underlying model identifier. If None, DEFAULT_MODEL is used.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set in the environment.")

    client = Groq(api_key=api_key)
    model_name = model_name or DEFAULT_MODEL

    user_content = _build_user_prompt(question, context)

    completion = client.chat.completions.create(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )

    message = completion.choices[0].message
    return message.content or ""

