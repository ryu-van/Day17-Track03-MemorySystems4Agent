from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    MemoryDecayStore,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
)
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B — advanced agent with three memory layers.

    1. Short-term (within-session) memory via CompactMemoryManager
    2. Persistent memory via User.md (UserProfileStore)
    3. Compact memory — older turns are summarized when token budget exceeded
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline

        # Persistent profile store wrapped with memory decay tracking
        _base_store = UserProfileStore(self.config.state_dir / "profiles")
        self.profile_store = MemoryDecayStore(profile_store=_base_store)

        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        # Map thread_id -> user_id for multi-user support
        self.thread_user: dict[str, str] = {}
        self.langchain_agent = None

        if not force_offline:
            self._maybe_build_langchain_agent()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route between offline and live modes."""
        self.thread_user[thread_id] = user_id

        if self.langchain_agent is not None:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    # ------------------------------------------------------------------
    # Offline path
    # ------------------------------------------------------------------

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic advanced path exercising all three memory layers.

        Bonus features active:
        - Confidence threshold: extract_profile_updates() filters low-confidence facts
        - Memory decay: upsert goes through MemoryDecayStore which tracks age/mentions
        - Conflict handling: upsert_fact overwrites old value and logs the change
        """

        # 1. Extract stable profile facts (with confidence threshold) and persist
        updates = extract_profile_updates(message)
        for key, value in updates.items():
            old_facts = self.profile_store.facts(user_id)
            old_value = old_facts.get(key)
            self.profile_store.upsert_fact(user_id, key, value)
            # Conflict handling: log when a correction replaces an old fact
            if old_value and old_value != value and not old_value.startswith("~"):
                # The new value is different — this is a correction.
                # upsert_fact already overwrites; we just note it in compact memory
                # so the agent is aware of the update during this session.
                conflict_note = f"[Correction detected: '{key}' changed from '{old_value}' to '{value}']"
                self.compact_memory.append(thread_id, "system", conflict_note)

        # 2. Append message to compact memory
        self.compact_memory.append(thread_id, "user", message)

        # 3. Estimate prompt context load
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )

        # 4. Generate response using persisted memory (with decay awareness)
        response_text = self._offline_response(user_id, thread_id, message)

        # 5. Append assistant reply to compact memory
        self.compact_memory.append(thread_id, "assistant", response_text)

        # 6. Track output tokens
        output_tokens = estimate_tokens(response_text)
        self.thread_tokens[thread_id] = (
            self.thread_tokens.get(thread_id, 0) + output_tokens
        )

        return {
            "response": response_text,
            "agent_tokens": output_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Estimate tokens carried into one turn.

        Includes: User.md content + compact summary + recent kept messages.
        """
        profile_text = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)
        summary: str = ctx.get("summary", "")  # type: ignore[assignment]
        messages: list[dict[str, str]] = ctx.get("messages", [])  # type: ignore[assignment]

        total = estimate_tokens(profile_text)
        total += estimate_tokens(summary)
        for msg in messages:
            total += estimate_tokens(msg.get("content", ""))
        return total

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Return a deterministic answer using all available persisted memory.

        Uses decayed_facts() so stale/unconfirmed facts are marked with '~'
        and handled with lower priority in responses.
        """
        # Use decay-aware facts: stale values have a "~" prefix
        facts = self.profile_store.decayed_facts(user_id)
        lower = message.lower()

        # Helper: check if question asks about specific facts
        def has(keywords: list[str]) -> bool:
            return any(kw in lower for kw in keywords)

        # Build a comprehensive recall answer when asked for a summary
        if has(["nhắc lại", "tóm tắt", "mô tả", "bạn biết", "nhớ lại"]):
            return self._build_summary_response(facts, message)

        # Specific fact questions
        if has(["tên"]) and "name" in facts:
            name = facts["name"]
            extras = []
            if "profession" in facts and has(["nghề", "làm gì", "công việc"]):
                extras.append(f"làm {facts['profession']}")
            if extras:
                return f"Bạn tên {name}, hiện {', '.join(extras)}."
            return f"Bạn tên là {name}."

        if has(["ở đâu", "nơi ở", "sống ở", "đang ở"]) and "location" in facts:
            return f"Hiện tại bạn đang ở {facts['location']}."

        if has(["nghề", "làm gì", "công việc", "chức vụ", "engineer", "mlops", "backend"]):
            if "profession" in facts:
                loc = facts.get("location", "")
                loc_text = f" tại {loc}" if loc else ""
                return f"Nghề nghiệp hiện tại của bạn là {facts['profession']}{loc_text}."
            return "Mình chưa có thông tin về nghề nghiệp của bạn."

        if has(["đồ uống", "uống gì", "cà phê", "thức uống"]):
            if "drink" in facts:
                return f"Đồ uống yêu thích của bạn là {facts['drink']}."
            return "Mình chưa có thông tin về đồ uống yêu thích của bạn."

        if has(["món ăn", "ăn gì", "thức ăn", "mì quảng"]):
            if "food" in facts:
                return f"Món ăn yêu thích của bạn là {facts['food']}."
            return "Mình chưa có thông tin về món ăn yêu thích của bạn."

        if has(["nuôi", "thú cưng", "corgi", "chó", "mèo"]):
            if "pet" in facts:
                return f"Bạn nuôi {facts['pet']}."
            return "Mình chưa có thông tin về thú cưng của bạn."

        if has(["style", "trả lời", "kiểu trả lời", "cách trả lời", "phong cách"]):
            if "response_style" in facts:
                return f"Style trả lời bạn thích: {facts['response_style']}."
            return "Bạn chưa đề cập cụ thể về style trả lời mong muốn."

        # Check compact memory context for recent conversation references
        ctx = self.compact_memory.context(thread_id)
        summary = ctx.get("summary", "")

        # Generic response acknowledging stored info
        if facts:
            stored = ", ".join(f"{k}: {v}" for k, v in list(facts.items())[:3])
            return (
                f"Mình đã ghi nhận thông tin của bạn ({stored}...). "
                f"Có điều gì cụ thể bạn muốn mình nhắc lại không?"
            )

        return (
            "Mình đã ghi nhận thông tin trong cuộc trò chuyện này. "
            "Bạn có thể hỏi mình về tên, nơi ở, nghề nghiệp hoặc sở thích đã đề cập."
        )

    def _build_summary_response(self, facts: dict[str, str], message: str) -> str:
        """Build a comprehensive profile summary from stored facts."""
        lower = message.lower()
        parts: list[str] = []

        # Determine which facts are relevant to the question
        want_name = any(kw in lower for kw in ["tên", "name"])
        want_location = any(kw in lower for kw in ["ở đâu", "nơi ở", "đang ở"])
        want_profession = any(kw in lower for kw in ["nghề", "làm gì", "công việc"])
        want_drink = any(kw in lower for kw in ["đồ uống", "uống", "cà phê"])
        want_food = any(kw in lower for kw in ["món ăn", "ăn"])
        want_style = any(kw in lower for kw in ["style", "trả lời", "phong cách"])
        want_pet = any(kw in lower for kw in ["nuôi", "thú cưng", "corgi"])

        # If no specific topic mentioned, show all
        want_all = not any([want_name, want_location, want_profession,
                            want_drink, want_food, want_style, want_pet])

        if (want_all or want_name) and "name" in facts:
            parts.append(f"**Tên**: {facts['name']}")
        if (want_all or want_location) and "location" in facts:
            parts.append(f"**Nơi ở**: {facts['location']}")
        if (want_all or want_profession) and "profession" in facts:
            parts.append(f"**Nghề nghiệp**: {facts['profession']}")
        if (want_all or want_drink) and "drink" in facts:
            parts.append(f"**Đồ uống yêu thích**: {facts['drink']}")
        if (want_all or want_food) and "food" in facts:
            parts.append(f"**Món ăn yêu thích**: {facts['food']}")
        if (want_all or want_pet) and "pet" in facts:
            parts.append(f"**Thú cưng**: {facts['pet']}")
        if (want_all or want_style) and "response_style" in facts:
            parts.append(f"**Style trả lời**: {facts['response_style']}")

        # Add any other stored facts
        if want_all:
            for k, v in facts.items():
                if k not in ("name", "location", "profession", "drink", "food", "pet", "response_style"):
                    parts.append(f"**{k}**: {v}")

        if not parts:
            return "Mình chưa có thông tin nào được lưu cho bạn."

        return "Thông tin mình đang lưu về bạn:\n" + "\n".join(f"- {p}" for p in parts)

    # ------------------------------------------------------------------
    # Live path (LangChain / LangGraph)
    # ------------------------------------------------------------------

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Call the live LangGraph agent with User.md + compact summary injected.

        Design: each turn uses a fresh LangGraph invocation (no checkpointer
        history replay). Instead, short-term context is managed by our own
        CompactMemoryManager, whose bounded summary + recent messages are
        injected into the system prompt each turn.

        This prevents LangGraph InMemorySaver from replaying the full
        conversation history O(n²) — the root cause of the 71k prompt token
        problem in the previous version.
        """
        # 1. Extract and persist profile facts
        updates = extract_profile_updates(message)
        for key, value in updates.items():
            old_facts = self.profile_store.facts(user_id)
            old_value = old_facts.get(key)
            self.profile_store.upsert_fact(user_id, key, value)
            if old_value and old_value != value and not old_value.startswith("~"):
                conflict_note = f"[Correction: '{key}' updated from '{old_value}' to '{value}']"
                self.compact_memory.append(thread_id, "system", conflict_note)

        # 2. Append message to our compact memory BEFORE reading context
        self.compact_memory.append(thread_id, "user", message)

        # 3. Build bounded context: User.md + compact summary + recent messages
        profile_text = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)
        summary: str = ctx.get("summary", "")  # type: ignore[assignment]
        recent_msgs: list[dict[str, str]] = ctx.get("messages", [])  # type: ignore[assignment]

        # Format recent messages as readable history (exclude the current user turn)
        history_lines = []
        for m in recent_msgs[:-1]:  # exclude last (current) message
            role = "User" if m["role"] == "user" else "Assistant"
            history_lines.append(f"{role}: {m['content'][:300]}")
        history_text = "\n".join(history_lines)

        system_prompt = (
            "You are a helpful assistant with persistent memory.\n\n"
            f"## User Profile (User.md)\n{profile_text}\n\n"
        )
        if summary:
            system_prompt += f"## Earlier Conversation Summary\n{summary}\n\n"
        if history_text:
            system_prompt += f"## Recent Messages\n{history_text}\n\n"
        system_prompt += (
            "Use the profile and context above to answer accurately. "
            "When the user shares new personal info, call upsert_user_fact to update."
        )

        # Estimate prompt tokens from what we actually inject
        prompt_tokens = (
            estimate_tokens(profile_text)
            + estimate_tokens(summary)
            + estimate_tokens(history_text)
            + estimate_tokens(message)
        )

        try:
            # Use a unique turn-scoped thread_id so LangGraph never replays history
            turn_thread_id = f"{thread_id}__turn_{self.thread_tokens.get(thread_id, 0)}"
            result = self.langchain_agent.invoke(
                {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": message},
                    ],
                },
                config={"configurable": {"thread_id": turn_thread_id}},
            )
            output = result.get("messages", [])
            response_text = output[-1].content if output else ""

            usage = getattr(output[-1], "usage_metadata", {}) if output else {}
            output_tokens = (usage.get("output_tokens") if usage else None) or estimate_tokens(response_text)

        except Exception as exc:
            response_text = f"[Live agent error: {exc}]"
            output_tokens = estimate_tokens(response_text)

        # 4. Append assistant reply to compact memory
        self.compact_memory.append(thread_id, "assistant", response_text)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + output_tokens
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens

        return {
            "response": response_text,
            "agent_tokens": output_tokens,
            "prompt_tokens": prompt_tokens,
        }

    # ------------------------------------------------------------------
    # LangChain agent builder (optional)
    # ------------------------------------------------------------------

    def _maybe_build_langchain_agent(self) -> None:
        """Wire a LangGraph agent with User.md tools and InMemorySaver."""
        try:
            from langchain_core.tools import tool
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.prebuilt import create_react_agent

            profile_store = self.profile_store

            @tool
            def read_user_profile(user_id: str) -> str:
                """Read the persistent User.md profile for a user."""
                return profile_store.read_text(user_id)

            @tool
            def write_user_profile(user_id: str, content: str) -> str:
                """Write the full content of User.md for a user."""
                profile_store.write_text(user_id, content)
                return f"Profile updated for {user_id}"

            @tool
            def upsert_user_fact(user_id: str, key: str, value: str) -> str:
                """Insert or update a single fact in User.md."""
                profile_store.upsert_fact(user_id, key, value)
                return f"Fact '{key}' updated to '{value}' for {user_id}"

            llm = build_chat_model(self.config.model)
            memory = MemorySaver()

            system_prompt = (
                "You are a helpful assistant with persistent memory. "
                "You have access to tools to read and write user profile information. "
                "Always check the user profile before answering recall questions. "
                "When the user shares personal information, update their profile."
            )

            self.langchain_agent = create_react_agent(
                llm,
                tools=[read_user_profile, write_user_profile, upsert_user_fact],
                checkpointer=memory,
                prompt=system_prompt,
            )
        except Exception:
            self.langchain_agent = None
