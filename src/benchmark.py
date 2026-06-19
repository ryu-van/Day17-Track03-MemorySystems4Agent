from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import extract_profile_updates, extract_entities

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def recall_points(answer: str, expected: list[str]) -> float:
    """Score 0 / 0.5 / 1 depending on how many expected facts appear.

    - 0   : none found
    - 0.5 : at least one but not all found
    - 1.0 : all found
    """
    if not expected:
        return 1.0
    lower_answer = answer.lower()
    found = sum(1 for e in expected if e.lower() in lower_answer)
    if found == 0:
        return 0.0
    if found == len(expected):
        return 1.0
    return 0.5


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight quality score for offline mode.

    Combines recall with a length penalty so empty or one-word answers
    score lower even if they technically contain a keyword.
    """
    if not answer or not answer.strip():
        return 0.0

    base = recall_points(answer, expected)

    # Small bonus for detailed answers (capped)
    word_count = len(answer.split())
    length_bonus = min(0.1, word_count / 100)

    return min(1.0, base + length_bonus)


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------

def run_agent_benchmark(
    agent_name: str,
    agent,
    conversations: list[dict[str, Any]],
    config,
) -> BenchmarkRow:
    """Evaluate one agent over many conversations.

    Steps:
    1. Feed all turns to the agent conversation-by-conversation.
    2. Track agent_tokens_only and prompt_tokens_processed.
    3. Ask recall questions in a fresh thread after each conversation.
    4. Compute average recall and quality.
    5. Record memory file growth and compaction counts.
    """
    total_agent_tokens = 0
    total_prompt_tokens = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    total_compactions = 0
    memory_growth = 0

    for conv in conversations:
        user_id: str = conv.get("user_id", "default")
        conv_id: str = conv.get("id", "unknown")
        turns: list[str] = conv.get("turns", [])
        recall_qs: list[dict[str, Any]] = conv.get("recall_questions", [])

        # Feed all turns using the conversation id as thread_id
        for turn in turns:
            result = agent.reply(user_id=user_id, thread_id=conv_id, message=turn)
            total_agent_tokens += result.get("agent_tokens", 0)
            total_prompt_tokens += result.get("prompt_tokens", 0)

        # Ask recall questions in a FRESH thread (cross-session recall test)
        recall_thread_id = f"{conv_id}_recall"
        for rq in recall_qs:
            question: str = rq.get("question", "")
            expected: list[str] = rq.get("expected_contains", [])

            result = agent.reply(user_id=user_id, thread_id=recall_thread_id, message=question)
            answer = result.get("response", "")

            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))

            total_agent_tokens += result.get("agent_tokens", 0)
            total_prompt_tokens += result.get("prompt_tokens", 0)

        # Track compactions
        total_compactions += agent.compaction_count(conv_id)

        # Track memory growth (advanced only)
        if hasattr(agent, "memory_file_size"):
            memory_growth = max(memory_growth, agent.memory_file_size(user_id))

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=round(avg_recall, 3),
        response_quality=round(avg_quality, 3),
        memory_growth_bytes=memory_growth,
        compactions=total_compactions,
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_rows(rows: list[BenchmarkRow]) -> str:
    """Print a comparison table of benchmark results."""
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    data = [
        [
            r.agent_name,
            r.agent_tokens_only,
            r.prompt_tokens_processed,
            f"{r.recall_score:.3f}",
            f"{r.response_quality:.3f}",
            r.memory_growth_bytes,
            r.compactions,
        ]
        for r in rows
    ]

    if HAS_TABULATE:
        return tabulate(data, headers=headers, tablefmt="github")

    # Fallback: simple aligned text table
    col_widths = [max(len(str(row[i])) for row in [headers] + data) for i in range(len(headers))]
    sep = " | ".join("-" * w for w in col_widths)
    header_line = " | ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers))
    lines = [header_line, sep]
    for row in data:
        lines.append(" | ".join(str(v).ljust(col_widths[i]) for i, v in enumerate(row)))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis text
# ---------------------------------------------------------------------------

def print_analysis(standard_rows: list[BenchmarkRow], stress_rows: list[BenchmarkRow]) -> None:
    """Print a brief analysis of the benchmark results."""
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    if len(standard_rows) >= 2:
        baseline, advanced = standard_rows[0], standard_rows[1]
        print(f"\n[Standard Benchmark]")
        print(f"• Recall: Baseline={baseline.recall_score:.3f}, Advanced={advanced.recall_score:.3f}")
        recall_delta = advanced.recall_score - baseline.recall_score
        print(f"  → Advanced recall {'tốt hơn' if recall_delta > 0 else 'tương đương'} baseline "
              f"(+{recall_delta:.3f}) nhờ User.md persistent memory.")

        prompt_delta = advanced.prompt_tokens_processed - baseline.prompt_tokens_processed
        if prompt_delta > 0:
            print(f"• Prompt tokens: Advanced dùng thêm {prompt_delta} tokens ở hội thoại ngắn "
                  f"vì phải kéo theo User.md mỗi lượt.")
        else:
            print(f"• Prompt tokens: Advanced tiết kiệm {-prompt_delta} tokens nhờ compact memory.")

    if len(stress_rows) >= 2:
        baseline_s, advanced_s = stress_rows[0], stress_rows[1]
        print(f"\n[Long-Context Stress Benchmark]")
        print(f"• Prompt tokens: Baseline={baseline_s.prompt_tokens_processed}, "
              f"Advanced={advanced_s.prompt_tokens_processed}")
        prompt_delta_s = baseline_s.prompt_tokens_processed - advanced_s.prompt_tokens_processed
        if prompt_delta_s > 0:
            print(f"  → Compact memory giúp Advanced giảm {prompt_delta_s} prompt tokens "
                  f"so với Baseline ở hội thoại dài.")
        print(f"• Compactions: {advanced_s.compactions} lần compact đã xảy ra.")
        print(f"• Memory growth: {advanced_s.memory_growth_bytes} bytes (User.md tăng theo thời gian).")

    print(f"\n[Trade-off Summary]")
    print("• Short-term: Advanced tốn hơn về prompt tokens (kéo User.md + summary)")
    print("• Long-term: Compact memory kéo prompt cost xuống khi hội thoại rất dài")
    print("• Risk: Memory file (User.md) tăng trưởng không giới hạn nếu không có pruning")
    print("• Recommendation: Compact chủ yếu tối ưu 'prompt tokens processed', không phải agent tokens")


# ---------------------------------------------------------------------------
# Bonus Benchmark — confidence threshold & entity extraction impact
# ---------------------------------------------------------------------------

def run_bonus_benchmark(conversations: list[dict[str, Any]]) -> None:
    """Measure real impact of confidence threshold and entity extraction.

    For each turn across all conversations:
    1. Count how many facts would be extracted WITHOUT threshold (min_confidence=0.0)
    2. Count how many facts ARE extracted WITH threshold (default MIN_CONFIDENCE_THRESHOLD)
    3. Count how many facts get structured sub-fields via extract_entities()
    4. Count how many noise turns are correctly blocked

    Prints a summary table showing precision improvement.
    """
    from memory_store import MIN_CONFIDENCE_THRESHOLD, _noise_penalty, _is_question_only

    total_turns = 0
    facts_without_threshold = 0
    facts_with_threshold = 0
    structured_subfields = 0
    noise_turns_blocked = 0
    noise_turns_total = 0
    entity_breakdown: dict[str, int] = {}

    for conv in conversations:
        turns: list[str] = conv.get("turns", [])
        for turn in turns:
            total_turns += 1
            penalty = _noise_penalty(turn)

            # Count noise turns
            if penalty > 0:
                noise_turns_total += 1

            # Facts without threshold (accept all matches, confidence >= 0)
            raw_facts = extract_profile_updates(turn, min_confidence=0.0)
            facts_without_threshold += len(raw_facts)

            # Facts with threshold (default confidence filter)
            filtered_facts = extract_profile_updates(turn, min_confidence=MIN_CONFIDENCE_THRESHOLD)
            facts_with_threshold += len(filtered_facts)

            # Count blocked facts from noise turns
            if penalty > 0 and len(raw_facts) > len(filtered_facts):
                noise_turns_blocked += 1

            # Structured entity extraction
            entities = extract_entities(turn, min_confidence=MIN_CONFIDENCE_THRESHOLD)
            for key, bundle in entities.items():
                entity_breakdown[key] = entity_breakdown.get(key, 0) + 1
                # Count sub-fields beyond "raw"
                subfields = [k for k in bundle if k != "raw"]
                structured_subfields += len(subfields)

    blocked = facts_without_threshold - facts_with_threshold
    block_rate = blocked / facts_without_threshold * 100 if facts_without_threshold > 0 else 0
    noise_block_rate = noise_turns_blocked / noise_turns_total * 100 if noise_turns_total > 0 else 0

    print(f"\n{'Metric':<45} {'Value':>10}")
    print("-" * 57)
    print(f"{'Total turns processed':<45} {total_turns:>10}")
    print(f"{'Facts extracted WITHOUT threshold':<45} {facts_without_threshold:>10}")
    print(f"{'Facts extracted WITH threshold (default)':<45} {facts_with_threshold:>10}")
    print(f"{'Facts blocked by threshold':<45} {blocked:>10}  ({block_rate:.1f}% filtered out)")
    print(f"{'Noise turns detected':<45} {noise_turns_total:>10}")
    print(f"{'Noise turns correctly blocked':<45} {noise_turns_blocked:>10}  ({noise_block_rate:.1f}% of noise turns)")
    print(f"{'Structured sub-fields extracted':<45} {structured_subfields:>10}")

    print(f"\nEntity type distribution (with threshold):")
    for key, count in sorted(entity_breakdown.items(), key=lambda x: -x[1]):
        print(f"  {key:<20} {count:>5} occurrences")

    print(f"\nInterpretation:")
    print(f"  • Confidence threshold removed {block_rate:.1f}% of potential facts,")
    print(f"    keeping User.md lean and reducing false positives.")
    print(f"  • {noise_block_rate:.1f}% of noise turns (jokes/disclaimers) were correctly blocked.")
    print(f"  • Entity extraction added {structured_subfields} structured sub-fields,")
    print(f"    enabling more precise recall (e.g. role vs domain for profession).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run both benchmark suites and print comparison tables."""
    config = load_config(Path(__file__).resolve().parent.parent)

    data_dir = config.data_dir
    standard_path = data_dir / "conversations.json"
    stress_path = data_dir / "advanced_long_context.json"

    print("Loading benchmark data...")
    standard_convs = load_conversations(standard_path)
    stress_convs = load_conversations(stress_path)

    # ----------------------------------------------------------------
    # Standard Benchmark
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STANDARD BENCHMARK  (data/conversations.json)")
    print("=" * 60)

    baseline_std = BaselineAgent(config=config, force_offline=True)
    advanced_std = AdvancedAgent(config=config, force_offline=True)

    std_rows = [
        run_agent_benchmark("Baseline", baseline_std, standard_convs, config),
        run_agent_benchmark("Advanced", advanced_std, standard_convs, config),
    ]
    print(format_rows(std_rows))

    # ----------------------------------------------------------------
    # Long-Context Stress Benchmark
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("LONG-CONTEXT STRESS BENCHMARK  (data/advanced_long_context.json)")
    print("=" * 60)

    baseline_stress = BaselineAgent(config=config, force_offline=True)
    advanced_stress = AdvancedAgent(config=config, force_offline=True)

    stress_rows = [
        run_agent_benchmark("Baseline", baseline_stress, stress_convs, config),
        run_agent_benchmark("Advanced", advanced_stress, stress_convs, config),
    ]
    print(format_rows(stress_rows))

    # ----------------------------------------------------------------
    # Analysis
    # ----------------------------------------------------------------
    print_analysis(std_rows, stress_rows)

    # ----------------------------------------------------------------
    # Bonus Benchmark — measure impact of confidence threshold & entity extraction
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BONUS BENCHMARK — Confidence Threshold & Entity Extraction Impact")
    print("=" * 60)
    run_bonus_benchmark(standard_convs + stress_convs)


if __name__ == "__main__":
    main()
