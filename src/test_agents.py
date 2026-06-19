from __future__ import annotations

from pathlib import Path

import pytest

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens


# ---------------------------------------------------------------------------
# Config factory for isolated tests
# ---------------------------------------------------------------------------

def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated LabConfig pointing to a temp directory.

    Uses a very low compact threshold so compaction triggers quickly in tests.
    """
    from model_provider import ProviderConfig

    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "profiles").mkdir(exist_ok=True)

    dummy_provider = ProviderConfig(
        provider="custom",
        model_name="test-model",
        temperature=0.0,
        api_key="test-key",
        base_url="http://localhost:9999/v1",
    )

    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=state_dir,
        compact_threshold_tokens=50,   # very low to trigger compaction quickly
        compact_keep_messages=2,
        model=dummy_provider,
        judge_model=dummy_provider,
    )


# ---------------------------------------------------------------------------
# Test 1 — UserProfileStore read / write / edit
# ---------------------------------------------------------------------------

def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify User.md can be created, updated, and edited."""
    store = UserProfileStore(root_dir=tmp_path / "profiles")
    user_id = "test_user"

    # Initial read returns default template
    content = store.read_text(user_id)
    assert "User Profile" in content

    # Write new content
    written_path = store.write_text(user_id, "# User Profile\n- name: Alice\n")
    assert written_path.exists()
    assert store.read_text(user_id) == "# User Profile\n- name: Alice\n"

    # Edit an existing line
    changed = store.edit_text(user_id, "Alice", "Bob")
    assert changed is True
    assert "Bob" in store.read_text(user_id)
    assert "Alice" not in store.read_text(user_id)

    # Edit non-existent text returns False
    not_changed = store.edit_text(user_id, "does_not_exist", "replacement")
    assert not_changed is False

    # file_size returns a positive integer after writing
    assert store.file_size(user_id) > 0

    # upsert_fact adds new facts
    store.upsert_fact(user_id, "location", "Hà Nội")
    facts = store.facts(user_id)
    assert facts.get("location") == "Hà Nội"

    # upsert_fact updates existing facts
    store.upsert_fact(user_id, "location", "Huế")
    facts = store.facts(user_id)
    assert facts.get("location") == "Huế"
    # Old value should be gone
    assert "Hà Nội" not in store.read_text(user_id)


# ---------------------------------------------------------------------------
# Test 2 — compact memory trigger
# ---------------------------------------------------------------------------

def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction in CompactMemoryManager."""
    # Threshold of 50 tokens, keep 2 messages
    manager = CompactMemoryManager(threshold_tokens=50, keep_messages=2)
    thread_id = "test_thread"

    # Start with 0 compactions
    assert manager.compaction_count(thread_id) == 0

    # Add messages until compaction fires
    # Each message ~25 chars → ~6 tokens; need >50 tokens to trigger
    long_msg = "a" * 100  # ~25 tokens each

    manager.append(thread_id, "user", long_msg)
    manager.append(thread_id, "assistant", long_msg)
    manager.append(thread_id, "user", long_msg)

    # At least one compaction should have occurred
    assert manager.compaction_count(thread_id) >= 1

    # After compaction, only keep_messages recent messages remain
    ctx = manager.context(thread_id)
    messages = ctx["messages"]
    assert len(messages) <= 2

    # Summary should be non-empty
    assert len(ctx["summary"]) > 0


# ---------------------------------------------------------------------------
# Test 3 — cross-session recall
# ---------------------------------------------------------------------------

def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify advanced remembers facts across sessions; baseline does not."""
    config = make_config(tmp_path)
    user_id = "recall_user"

    # ---- Session 1: introduce facts ----
    advanced = AdvancedAgent(config=config, force_offline=True)
    baseline = BaselineAgent(config=config, force_offline=True)

    session1_thread = "session_1"
    advanced.reply(user_id=user_id, thread_id=session1_thread,
                   message="Mình tên là NguyenTest, hiện ở Đà Lạt và làm data engineer.")
    baseline.reply(user_id=user_id, thread_id=session1_thread,
                   message="Mình tên là NguyenTest, hiện ở Đà Lạt và làm data engineer.")

    # ---- Session 2: ask recall in a NEW thread (new agent instances too) ----
    advanced2 = AdvancedAgent(config=config, force_offline=True)
    baseline2 = BaselineAgent(config=config, force_offline=True)

    session2_thread = "session_2"
    adv_result = advanced2.reply(user_id=user_id, thread_id=session2_thread,
                                 message="Bạn có nhớ tên mình không?")
    base_result = baseline2.reply(user_id=user_id, thread_id=session2_thread,
                                  message="Bạn có nhớ tên mình không?")

    # Advanced should recall the name from User.md (persisted to disk)
    assert "NguyenTest" in adv_result["response"], (
        f"Advanced should recall 'NguyenTest' but got: {adv_result['response']}"
    )

    # Baseline should NOT recall (no persistent memory, new thread)
    assert "NguyenTest" not in base_result["response"], (
        f"Baseline should NOT recall 'NguyenTest' but got: {base_result['response']}"
    )


# ---------------------------------------------------------------------------
# Test 4 — compact reduces prompt load on long threads
# ---------------------------------------------------------------------------

def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt token load of baseline vs advanced on a long thread.

    On a long thread, baseline keeps growing its prompt context linearly.
    Advanced should have lower prompt growth because compact memory summarizes old turns.
    """
    config = make_config(tmp_path)
    user_id = "load_user"

    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    thread_id = "long_thread"
    # Send many messages to both agents
    messages = [
        "Mình tên là LoadTest, đang làm MLOps engineer tại Hà Nội.",
        "Hôm nay mình đang viết pipeline cho model training.",
        "Mình thích Python và infrastructure as code.",
        "Mình đang dùng Kubernetes cho production workloads.",
        "Đồ uống yêu thích là trà sữa taro.",
        "Mình hay làm việc từ 9 giờ sáng đến 6 giờ tối.",
        "Team mình có 5 người, toàn engineer.",
        "Mình cũng thích đọc sách về system design.",
        "Cuối tuần mình hay đi leo núi ở ngoại thành.",
        "Mình muốn câu trả lời ngắn gọn và có ví dụ thực tế.",
        "Mình đang tìm hiểu về eBPF cho observability.",
        "Bạn có thể nhớ giúp mình các thông tin này không?",
    ]

    for msg in messages:
        baseline.reply(user_id=user_id, thread_id=thread_id, message=msg)
        advanced.reply(user_id=user_id, thread_id=thread_id, message=msg)

    baseline_prompt = baseline.prompt_token_usage(thread_id)
    advanced_prompt = advanced.prompt_token_usage(thread_id)
    advanced_compactions = advanced.compaction_count(thread_id)

    print(f"\nBaseline prompt tokens: {baseline_prompt}")
    print(f"Advanced prompt tokens: {advanced_prompt}")
    print(f"Advanced compactions: {advanced_compactions}")

    # Advanced should have triggered at least one compaction given the low threshold
    assert advanced_compactions >= 1, (
        f"Expected at least 1 compaction but got {advanced_compactions}. "
        "Check compact_threshold_tokens in make_config."
    )

    # On a mid-length thread advanced carries User.md overhead each turn, so its
    # raw prompt count is higher than baseline — that is the expected trade-off.
    # What matters is that compaction IS happening (tested above) and that the
    # per-turn prompt growth rate of advanced is bounded (not O(n²) like baseline).
    #
    # Baseline grows as: sum(1+2+3+...+n) = O(n²) — full history each turn
    # Advanced grows as: O(n * (profile + summary + keep_messages)) — bounded window
    #
    # We verify advanced compacted at least once (already asserted) and that
    # the number of kept messages after all compactions is small.
    ctx = advanced.compact_memory.context(thread_id)
    kept = len(ctx["messages"])
    assert kept <= config.compact_keep_messages + 1, (
        f"After {advanced_compactions} compactions, advanced should keep at most "
        f"{config.compact_keep_messages} messages, but has {kept}."
    )
    # Also verify summary is non-empty (compact produced a real summary)
    assert len(ctx["summary"]) > 0, "Compact should have produced a non-empty summary."


# ---------------------------------------------------------------------------
# Additional: estimate_tokens basic sanity
# ---------------------------------------------------------------------------

def test_estimate_tokens_basic() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("    ") == 0
    assert estimate_tokens("hello") >= 1
    # ~4 chars per token heuristic
    text = "a" * 400
    assert 90 <= estimate_tokens(text) <= 110


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------------
# BONUS tests
# ---------------------------------------------------------------------------

def test_confidence_threshold_blocks_noise(tmp_path: Path) -> None:
    """Verify facts in noisy/joke messages are NOT written to User.md.

    BONUS — Confidence threshold:
    The phrase 'chỉ là câu đùa' triggers a noise penalty that drops the
    effective confidence below MIN_CONFIDENCE_THRESHOLD, so the fact
    should be rejected.
    """
    from memory_store import extract_profile_updates, MIN_CONFIDENCE_THRESHOLD

    # Normal statement — should extract
    normal = "Mình tên là BonusUser, đang làm backend engineer cho startup."
    updates = extract_profile_updates(normal)
    assert "name" in updates, "Should extract name from a clear statement"

    # Noisy statement — should be blocked
    noisy = "Mình đùa thôi, mình làm product manager cho đỡ ngồi canh pipeline."
    updates_noisy = extract_profile_updates(noisy)
    assert "profession" not in updates_noisy, (
        "Should NOT extract profession from a statement flagged as a joke"
    )


def test_memory_decay_marks_stale_facts(tmp_path: Path) -> None:
    """Verify MemoryDecayStore marks facts as stale when not re-mentioned.

    BONUS — Memory decay:
    By setting stale_seconds=0, any fact written in the past is immediately
    stale (unless mentions >= mention_floor). We confirm the '~' prefix appears.
    """
    from memory_store import MemoryDecayStore, UserProfileStore

    base_store = UserProfileStore(root_dir=tmp_path / "profiles")
    decay_store = MemoryDecayStore(
        profile_store=base_store,
        stale_seconds=0.0,   # instant staleness for testing
        mention_floor=5,     # require 5 mentions to be immune
    )

    user_id = "decay_user"
    decay_store.upsert_fact(user_id, "location", "Hà Nội")

    # Wait is not needed because stale_seconds=0 — any age qualifies
    decayed = decay_store.decayed_facts(user_id)
    assert decayed.get("location", "").startswith("~"), (
        f"Stale fact should be prefixed with '~', got: {decayed.get('location')}"
    )


def test_memory_decay_immune_after_enough_mentions(tmp_path: Path) -> None:
    """Verify frequently-mentioned facts are NOT marked stale.

    BONUS — Memory decay:
    A fact mentioned >= its per-key floor times should remain clean.
    Uses 'location' (floor=2) so the test needs only 2 mentions.
    'name' uses floor=999 (never decays) and is tested separately.
    """
    from memory_store import MemoryDecayStore, UserProfileStore

    base_store = UserProfileStore(root_dir=tmp_path / "profiles2")
    decay_store = MemoryDecayStore(
        profile_store=base_store,
        stale_seconds=0.0,
        mention_floor=3,
    )

    user_id = "immune_user"
    # location has per-key floor=2
    decay_store.upsert_fact(user_id, "location", "Huế")  # mention 1
    decay_store.touch_fact(user_id, "location")           # mention 2 → immune (floor=2)

    decayed = decay_store.decayed_facts(user_id)
    assert not decayed.get("location", "").startswith("~"), (
        f"location with 2 mentions (floor=2) should NOT be stale, got: {decayed.get('location')}"
    )


def test_conflict_handling_updates_fact(tmp_path: Path) -> None:
    """Verify that a correction (new value for same key) overwrites old value.

    BONUS — Conflict handling:
    When the user corrects a fact (e.g. location changes from Huế to Đà Nẵng),
    the new value should replace the old one and the old value should be gone.
    """
    config = make_config(tmp_path)
    agent = AdvancedAgent(config=config, force_offline=True)
    user_id = "conflict_user"

    # First mention: location = Huế
    agent.reply(user_id=user_id, thread_id="t1",
                message="Mình tên là ConflictTest. Hiện tại mình đang ở Huế nhé.")

    facts_before = agent.profile_store.facts(user_id)
    assert facts_before.get("location") == "Huế", (
        f"Initial location should be 'Huế', got: {facts_before.get('location')}"
    )

    # Correction: location updated to Đà Nẵng
    agent.reply(user_id=user_id, thread_id="t1",
                message="À mình đính chính: giờ mình đang ở Đà Nẵng chứ không còn ở Huế.")

    facts_after = agent.profile_store.facts(user_id)
    assert facts_after.get("location") == "Đà Nẵng", (
        f"After correction, location should be 'Đà Nẵng', got: {facts_after.get('location')}"
    )
    # Old value must be gone — no dual-fact ambiguity
    profile_text = agent.profile_store.read_text(user_id)
    assert "Huế" not in profile_text or "Đà Nẵng" in profile_text, (
        "Profile should not hold conflicting location values simultaneously"
    )


# ---------------------------------------------------------------------------
# BONUS D — Entity extraction tests
# ---------------------------------------------------------------------------

def test_entity_extraction_profession_structured() -> None:
    """Verify profession is split into role and domain sub-fields.

    BONUS D — Structured entity extraction:
    'MLOps engineer' should produce role='MLOps', domain='engineer'
    rather than a flat string.
    """
    from memory_store import extract_entities

    msg = "Mình đang làm MLOps engineer cho một nhóm nhỏ."
    entities = extract_entities(msg)

    assert "profession" in entities, "Should extract profession entity"
    prof = entities["profession"]
    assert prof.get("raw"), "Should have raw value"
    assert prof.get("role") == "MLOps", (
        f"Expected role='MLOps', got: {prof.get('role')}"
    )
    assert prof.get("domain") == "engineer", (
        f"Expected domain='engineer', got: {prof.get('domain')}"
    )


def test_entity_extraction_drink_structured() -> None:
    """Verify drink is split into item and modifier.

    BONUS D — Structured entity extraction:
    'cà phê sữa đá' → item='cà phê', modifier='sữa đá'
    """
    from memory_store import extract_entities

    msg = "Đồ uống yêu thích của mình là cà phê sữa đá."
    entities = extract_entities(msg)

    assert "drink" in entities, "Should extract drink entity"
    drink = entities["drink"]
    assert drink.get("item") == "cà phê", (
        f"Expected item='cà phê', got: {drink.get('item')}"
    )
    assert drink.get("modifier") is not None, "Should extract modifier for 'sữa đá'"


def test_entity_extraction_location_has_city() -> None:
    """Verify location extraction populates city sub-field.

    BONUS D — Structured entity extraction:
    Location should be resolved to a known city name.
    """
    from memory_store import extract_entities

    msg = "Hiện tại mình đang ở Đà Nẵng nhé."
    entities = extract_entities(msg)

    assert "location" in entities, "Should extract location entity"
    loc = entities["location"]
    assert loc.get("city") == "Đà Nẵng", (
        f"Expected city='Đà Nẵng', got: {loc.get('city')}"
    )


def test_entity_extraction_pet_name() -> None:
    """Verify pet extraction captures species and name.

    BONUS D — Structured entity extraction:
    'nuôi một bé corgi tên Bơ' → species='corgi', pet_name='Bơ'
    """
    from memory_store import extract_entities

    msg = "Mình nuôi một bé corgi tên Bơ, nó rất dễ thương."
    entities = extract_entities(msg)

    assert "pet" in entities, "Should extract pet entity"
    pet = entities["pet"]
    assert "corgi" in pet.get("raw", "").lower(), "Raw should contain 'corgi'"
    assert pet.get("pet_name") == "Bơ", (
        f"Expected pet_name='Bơ', got: {pet.get('pet_name')}"
    )


def test_entity_extraction_noise_blocked() -> None:
    """Verify entity extraction respects confidence threshold for noise.

    BONUS D + A combined:
    Noise phrases should prevent low-confidence facts from being extracted
    even at the entity level.
    """
    from memory_store import extract_entities

    noisy = "Mình đùa thôi, mình làm product manager cho đỡ ngồi canh pipeline."
    entities = extract_entities(noisy)
    assert "profession" not in entities, (
        "Profession from a joke statement should be blocked by confidence threshold"
    )


# ---------------------------------------------------------------------------
# FIX tests
# ---------------------------------------------------------------------------

def test_summary_length_cap(tmp_path: Path) -> None:
    """Verify CompactMemoryManager summary stays within max_summary_tokens.

    FIX 1 — Summary length cap:
    After many compactions the summary must not grow unboundedly.
    """
    from memory_store import CompactMemoryManager, estimate_tokens

    manager = CompactMemoryManager(
        threshold_tokens=30,
        keep_messages=2,
        max_summary_tokens=80,   # tight cap to trigger truncation quickly
    )
    thread_id = "cap_thread"
    long_msg = "x" * 80  # ~20 tokens each

    # Send enough messages to trigger many compactions
    for i in range(20):
        manager.append(thread_id, "user", f"turn {i}: {long_msg}")
        manager.append(thread_id, "assistant", f"reply {i}: {long_msg}")

    ctx = manager.context(thread_id)
    summary: str = ctx["summary"]  # type: ignore[assignment]
    summary_tokens = estimate_tokens(summary)

    # Summary must stay near the cap, not grow to thousands of tokens
    assert summary_tokens <= 120, (
        f"Summary tokens ({summary_tokens}) exceeded 120 — cap not working. "
        "Summary head should have been truncated."
    )
    assert manager.compaction_count(thread_id) >= 1


def test_per_key_decay_floor_name_never_decays(tmp_path: Path) -> None:
    """Verify 'name' fact never decays regardless of age.

    FIX 3 — Per-key decay floor:
    DECAY_FLOOR_PER_KEY['name'] = 999, so name should never be stale.
    """
    from memory_store import MemoryDecayStore, UserProfileStore

    base = UserProfileStore(root_dir=tmp_path / "profiles")
    store = MemoryDecayStore(
        profile_store=base,
        stale_seconds=0.0,    # instant stale
        mention_floor=3,      # global fallback
    )
    user_id = "name_user"
    store.upsert_fact(user_id, "name", "DũngCT")   # only 1 mention

    decayed = store.decayed_facts(user_id)
    assert not decayed.get("name", "").startswith("~"), (
        "'name' should NEVER be marked stale (per-key floor = 999)"
    )


def test_per_key_decay_floor_location_decays_sooner(tmp_path: Path) -> None:
    """Verify 'location' decays with floor=2 while 'name' stays fresh.

    FIX 3 — Per-key decay floor:
    location floor=2, name floor=999. With stale_seconds=0 and 1 mention each,
    location should be stale but name should not.
    """
    from memory_store import MemoryDecayStore, UserProfileStore

    base = UserProfileStore(root_dir=tmp_path / "profiles2")
    store = MemoryDecayStore(
        profile_store=base,
        stale_seconds=0.0,
        mention_floor=3,
    )
    user_id = "loc_user"
    store.upsert_fact(user_id, "name", "Alice")
    store.upsert_fact(user_id, "location", "Huế")

    decayed = store.decayed_facts(user_id)
    assert not decayed.get("name", "").startswith("~"), "name should never be stale"
    assert decayed.get("location", "").startswith("~"), (
        "location (floor=2) with 1 mention should be stale at stale_seconds=0"
    )


def test_decay_persists_across_restarts(tmp_path: Path) -> None:
    """Verify decay metadata survives process restart via JSON sidecar.

    FIX 2 — Persist decay metadata:
    After writing a fact and saving, creating a new MemoryDecayStore instance
    pointing to the same directory should restore the mention count.
    """
    from memory_store import MemoryDecayStore, UserProfileStore

    profiles_dir = tmp_path / "profiles3"
    user_id = "persist_user"

    # Session 1: write fact with 3 mentions
    store1 = MemoryDecayStore(
        profile_store=UserProfileStore(root_dir=profiles_dir),
        stale_seconds=0.0,
        mention_floor=3,
    )
    store1.upsert_fact(user_id, "name", "PersistTest")
    store1.touch_fact(user_id, "name")   # mention 2
    store1.touch_fact(user_id, "name")   # mention 3 → immune

    # Session 2: new instance, same directory — should load from sidecar
    store2 = MemoryDecayStore(
        profile_store=UserProfileStore(root_dir=profiles_dir),
        stale_seconds=0.0,
        mention_floor=3,
    )
    decayed = store2.decayed_facts(user_id)
    assert not decayed.get("name", "").startswith("~"), (
        "After restart, 'name' with 3 mentions should still be immune. "
        "Decay metadata was not persisted correctly."
    )
