from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A — baseline with within-session memory only.

    - Remembers conversation within the same thread_id.
    - No persistent User.md.
    - Forgets everything when a new thread_id is used.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None

        if not force_offline:
            self._maybe_build_langchain_agent()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return agent response and token accounting."""
        if self.langchain_agent is not None:
            return self._reply_live(thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        """Return cumulative agent (output) token count for one thread."""
        session = self.sessions.get(thread_id)
        if session is None:
            return 0
        return session.token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        """Estimate total prompt context tokens processed across all turns."""
        session = self.sessions.get(thread_id)
        if session is None:
            return 0
        return session.prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        """Baseline has no compact memory — always 0."""
        return 0

    # ------------------------------------------------------------------
    # Offline path
    # ------------------------------------------------------------------

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic offline behavior for benchmarking without a live LLM."""
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()

        session = self.sessions[thread_id]

        # Append new user message
        session.messages.append({"role": "user", "content": message})

        # Estimate prompt tokens: all messages in this thread so far
        prompt_tokens = sum(
            estimate_tokens(m["content"]) for m in session.messages
        )
        session.prompt_tokens_processed += prompt_tokens

        # Generate a simple deterministic response
        response_text = self._generate_offline_response(session.messages, message)

        # Append assistant reply
        session.messages.append({"role": "assistant", "content": response_text})

        # Estimate output tokens
        output_tokens = estimate_tokens(response_text)
        session.token_usage += output_tokens

        return {
            "response": response_text,
            "agent_tokens": output_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _generate_offline_response(
        self, messages: list[dict[str, str]], last_message: str
    ) -> str:
        """Generate a simple offline response based on in-session history only.

        The baseline can only reference what was said in the current thread.
        It never has access to cross-session facts.
        """
        # Look for recall questions and try to answer from current session
        lower = last_message.lower()

        # Check if the question is about something mentioned earlier in this session
        facts_in_session: dict[str, str] = {}
        for msg in messages[:-1]:  # exclude current message
            if msg["role"] == "user":
                content = msg["content"]
                # Extract simple facts mentioned in conversation
                import re
                name_m = re.search(
                    r"(?:mình|tôi)\s+tên\s+(?:là\s+)?([A-Za-zÀ-ỹ0-9_]+(?:\s+[A-Za-zÀ-ỹ0-9]+)*)",
                    content, re.IGNORECASE
                )
                if name_m and "name" not in facts_in_session:
                    facts_in_session["name"] = name_m.group(1).strip()

                drink_m = re.search(
                    r"(?:đồ uống|thức uống)\s+yêu thích\s+(?:là\s+)?([^\.,]+)",
                    content, re.IGNORECASE
                )
                if drink_m and "drink" not in facts_in_session:
                    facts_in_session["drink"] = drink_m.group(1).strip()

        # Build contextual response
        if any(kw in lower for kw in ["tên", "name"]):
            if "name" in facts_in_session:
                return f"Trong cuộc trò chuyện này, bạn cho biết tên là {facts_in_session['name']}."
            return "Xin lỗi, trong phiên này bạn chưa cho mình biết tên."

        if any(kw in lower for kw in ["đồ uống", "uống", "cà phê"]):
            if "drink" in facts_in_session:
                return f"Đồ uống yêu thích bạn đề cập là {facts_in_session['drink']}."
            return "Xin lỗi, trong phiên này bạn chưa đề cập đồ uống yêu thích."

        # Generic acknowledgment for non-recall messages
        return (
            f"Mình đã ghi nhận thông tin trong phiên này. "
            f"(Baseline agent chỉ nhớ trong cùng thread, không có long-term memory.)"
        )

    # ------------------------------------------------------------------
    # Live path (LangChain / LangGraph)
    # ------------------------------------------------------------------

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        """Call the live LangChain agent and track token usage."""
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()

        session = self.sessions[thread_id]

        try:
            result = self.langchain_agent.invoke(
                {"messages": [{"role": "user", "content": message}]},
                config={"configurable": {"thread_id": thread_id}},
            )
            # Extract response text
            output = result.get("messages", [])
            response_text = output[-1].content if output else ""

            # Track tokens if available
            usage = result.get("usage_metadata", {})
            output_tokens = usage.get("output_tokens", estimate_tokens(response_text))
            prompt_tokens = usage.get("input_tokens", estimate_tokens(message))

        except Exception as exc:
            response_text = f"[Live agent error: {exc}]"
            output_tokens = estimate_tokens(response_text)
            prompt_tokens = estimate_tokens(message)

        session.messages.append({"role": "user", "content": message})
        session.messages.append({"role": "assistant", "content": response_text})
        session.token_usage += output_tokens
        session.prompt_tokens_processed += prompt_tokens

        return {
            "response": response_text,
            "agent_tokens": output_tokens,
            "prompt_tokens": prompt_tokens,
        }

    # ------------------------------------------------------------------
    # LangChain agent builder (optional)
    # ------------------------------------------------------------------

    def _maybe_build_langchain_agent(self) -> None:
        """Wire a LangGraph agent with InMemorySaver for within-session memory."""
        try:
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.prebuilt import create_react_agent

            llm = build_chat_model(self.config.model)
            memory = MemorySaver()

            system_prompt = (
                "You are a helpful assistant. "
                "You only remember what was said in this conversation. "
                "You do not have access to any information from previous sessions."
            )

            self.langchain_agent = create_react_agent(
                llm,
                tools=[],
                checkpointer=memory,
                prompt=system_prompt,
            )
        except Exception:
            # If LangChain/LangGraph is not installed or model fails, stay offline
            self.langchain_agent = None
