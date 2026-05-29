"""
chatbot.py
----------
Orchestrates the full RAG pipeline:
  1. Receive user message
  2. Retrieve context via HybridRetriever
  3. Build prompt with context + conversation history
  4. Call Claude API
  5. Return structured response

Memory: stores last N conversation turns so Claude can reference
        previous questions and answers in the same session.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import anthropic
from dotenv import load_dotenv
from src.retriever import HybridRetriever

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
MODEL         = "claude-sonnet-4-6"
MAX_TOKENS    = 1024
MEMORY_TURNS  = 5      # how many previous turns to include in context


# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a knowledgeable and friendly Japan travel assistant.
You help travelers plan their trips to Japan using information from the official
Japan National Tourism Organization (JNTO) website.

Guidelines:
- Answer based on the provided context only
- Be concise, helpful and enthusiastic about Japan
- If the context doesn't contain enough information, say so honestly
- Use specific details from the context when available
- Format lists and key points clearly when helpful"""


def build_prompt_qa_direct(question: str, answer: str, history: list) -> list:
    """
    Prompt for qa_direct mode.
    We already have a good answer — Claude refines and personalizes it.
    """
    history_text = _format_history(history)

    user_content = f"""{history_text}
        Retrieved answer for: "{question}"
        Answer from knowledge base: {answer}

        Using the retrieved answer above, provide a helpful, conversational response.
        You may expand slightly but stay faithful to the facts provided."""

    return [{"role": "user", "content": user_content}]


def build_prompt_vector_search(question: str, context: str, history: list) -> list:
    """
    Prompt for vector_search mode.
    Claude generates an answer from raw document chunks.
    """
    history_text = _format_history(history)

    user_content = f"""{history_text}
        Context from Japan Travel knowledge base:
        {context}

        Question: {question}

        Answer the question using only the context provided above.
        If the context doesn't fully answer the question, say what you do know
        and suggest the user visit japan.travel for more details."""

    return [{"role": "user", "content": user_content}]


def _format_history(history: list) -> str:
    """Format conversation history as readable text for the prompt."""
    if not history:
        return ""

    lines = ["Previous conversation:"]
    for turn in history:
        lines.append(f"User: {turn['user']}")
        lines.append(f"Assistant: {turn['assistant']}")
    lines.append("")   # blank line separator

    return "\n".join(lines)


# ── Chatbot class ─────────────────────────────────────────────────────────────

class JapanTravelChatbot:
    """
    Main chatbot class. Holds the retriever, Claude client,
    and conversation memory for a single session.

    Usage:
        bot = JapanTravelChatbot()
        response = bot.chat("What's the best time to visit Kyoto?")
        print(response["answer"])
        print(response["mode"])    # "qa_direct" or "vector_search"
        print(response["sources"]) # list of source pages
    """

    def __init__(self):
        print("🤖 Initializing Japan Travel Chatbot...")
        self.retriever = HybridRetriever()
        self.client    = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        self.history   = []   # list of {user, assistant} dicts
        print("✅ Chatbot ready!\n")

    def chat(self, user_message: str) -> dict:
        """
        Process a user message and return a structured response.

        Returns:
            {
                "answer":  str,          # Claude's response
                "mode":    str,          # "qa_direct" | "vector_search"
                "sources": list[dict],   # [{url, title, score}]
                "score":   float,        # top retrieval score
            }
        """
        # 1. Retrieve context
        retrieval = self.retriever.retrieve(user_message)

        # 2. Build prompt based on retrieval mode
        recent_history = self.history[-MEMORY_TURNS:]

        if retrieval["mode"] == "qa_direct":
            messages = build_prompt_qa_direct(
                question = user_message,
                answer   = retrieval["answer"],
                history  = recent_history,
            )
        else:
            messages = build_prompt_vector_search(
                question = user_message,
                context  = retrieval["context"],
                history  = recent_history,
            )

        # 3. Call Claude API
        response = self.client.messages.create(
            model     = MODEL,
            max_tokens= MAX_TOKENS,
            system    = SYSTEM_PROMPT,
            messages  = messages,
        )

        answer = response.content[0].text.strip()

        # 4. Update conversation memory
        self.history.append({
            "user":      user_message,
            "assistant": answer,
        })

        # 5. Return structured response
        return {
            "answer":  answer,
            "mode":    retrieval["mode"],
            "sources": retrieval["sources"],
            "score":   retrieval["score"],
            "matched_question": retrieval.get("matched_question"),
            "matched_answer": retrieval.get("answer") if retrieval["mode"] == "qa_direct" else None
        }

    def clear_history(self):
        """Reset conversation memory."""
        self.history = []
        print("🗑️  Conversation history cleared.")


# ── Terminal test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    bot = JapanTravelChatbot()

    test_questions = [
        "What airports serve Tokyo?",
        "What's the best time to visit Kyoto for cherry blossoms?",
        "Can you tell me more about the food scene there?",  # uses memory: "there" = Kyoto
    ]

    print("=" * 60)
    for question in test_questions:
        print(f"\n👤 User: {question}")
        response = bot.chat(question)
        print(f"🤖 Bot [{response['mode']} | score: {response['score']}]:")
        print(f"   {response['answer'][:300]}...")
        if response["sources"]:
            print(f"   📎 Source: {response['sources'][0]['title'][:50]}")
        print("-" * 60)