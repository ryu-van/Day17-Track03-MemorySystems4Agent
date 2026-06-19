from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple, TypedDict


# ---------------------------------------------------------------------------
# Token estimator
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Simple heuristic token estimator.

    Uses character count divided by 4 as a rough approximation.
    Good enough for offline benchmark comparisons.
    """
    if not text or not text.strip():
        return 0
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# UserProfileStore — persistent User.md management
# ---------------------------------------------------------------------------

_DEFAULT_PROFILE = """\
# User Profile

(no information yet)
"""


@dataclass
class UserProfileStore:
    """Persistent storage for User.md files, one per user_id."""

    root_dir: Path

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, user_id: str) -> Path:
        """Return the markdown file path for a given user_id.

        Sanitizes user_id so it is safe as a filename.
        """
        safe_id = re.sub(r"[^\w\-]", "_", user_id).strip("_") or "default"
        return self.root_dir / f"{safe_id}.md"

    def read_text(self, user_id: str) -> str:
        """Return file content, or a default empty profile if missing."""
        p = self.path_for(user_id)
        if p.exists():
            return p.read_text(encoding="utf-8")
        return _DEFAULT_PROFILE

    def write_text(self, user_id: str, content: str) -> Path:
        """Write markdown content to disk and return the file path."""
        p = self.path_for(user_id)
        p.write_text(content, encoding="utf-8")
        return p

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        """Replace the first occurrence of search_text in User.md.

        Returns True if a change was made, False otherwise.
        """
        current = self.read_text(user_id)
        if search_text not in current:
            return False
        updated = current.replace(search_text, replacement, 1)
        self.write_text(user_id, updated)
        return True

    def file_size(self, user_id: str) -> int:
        """Return current file size in bytes (0 if not yet created)."""
        p = self.path_for(user_id)
        if p.exists():
            return p.stat().st_size
        return 0

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        """Insert or update a fact line in User.md.

        Facts are stored as '- key: value' bullet lines.
        """
        current = self.read_text(user_id)

        # Remove default placeholder if present
        current = current.replace("(no information yet)", "").strip()

        pattern = re.compile(rf"^- {re.escape(key)}:.*$", re.MULTILINE)
        new_line = f"- {key}: {value}"

        if pattern.search(current):
            updated = pattern.sub(new_line, current)
        else:
            # Append under the header
            if "# User Profile" not in current:
                current = "# User Profile\n\n" + current
            updated = current.rstrip() + "\n" + new_line + "\n"

        self.write_text(user_id, updated)

    def facts(self, user_id: str) -> dict[str, str]:
        """Return all key/value facts stored in User.md."""
        text = self.read_text(user_id)
        result: dict[str, str] = {}
        for line in text.splitlines():
            m = re.match(r"^- ([\w\s]+):\s*(.+)$", line.strip())
            if m:
                result[m.group(1).strip()] = m.group(2).strip()
        return result


# ---------------------------------------------------------------------------
# BONUS A — Confidence threshold
# ---------------------------------------------------------------------------
# Each fact pattern now carries a confidence score (0.0–1.0).
# Only facts that meet MIN_CONFIDENCE_THRESHOLD are written to User.md.
# This prevents noisy or partial matches from polluting the profile.
#
# Design:
#   - High confidence (0.9): very explicit statement ("mình tên là X")
#   - Medium confidence (0.7): somewhat explicit ("mình ở X")
#   - Low confidence (0.5): inferred / indirect ("đang làm việc ở X")
#
# Risk introduced:
#   - Facts mentioned obliquely may never be stored.
#   - The threshold is a global constant; per-key tuning would be more precise.

MIN_CONFIDENCE_THRESHOLD: float = 0.65


class _PatternEntry(NamedTuple):
    pattern: str
    confidence: float


# Patterns ordered from most specific to most general.
# Each tuple: (fact_key, list_of_PatternEntry)
_FACT_PATTERNS: list[tuple[str, list[_PatternEntry]]] = [
    ("name", [
        _PatternEntry(r"(?:mình|tôi|tao)\s+tên\s+(?:là\s+)?([A-Za-zÀ-ỹ0-9_]+(?:\s+[A-Za-zÀ-ỹ0-9]+)*)", 0.95),
        _PatternEntry(r"tên\s+(?:của\s+)?(?:mình|tôi)\s+(?:là\s+)?([A-Za-zÀ-ỹ0-9_]+(?:\s+[A-Za-zÀ-ỹ0-9]+)*)", 0.90),
        _PatternEntry(r"(?:chào|hi|hello)[,.]?\s+(?:mình|tôi)\s+(?:là\s+|tên\s+)?([A-Za-zÀ-ỹ0-9_]+(?:\s+[A-Za-zÀ-ỹ0-9]+)*)", 0.80),
    ]),
    ("location", [
        _PatternEntry(r"(?:giờ|hiện tại|bây giờ|đang)\s+(?:mình|tôi)\s+(?:đang\s+)?ở\s+([A-Za-zÀ-ỹ\s]+?)(?:\s+(?:chứ|nhé|rồi|và|,|\.))", 0.90),
        _PatternEntry(r"(?:mình|tôi)\s+(?:đang\s+)?(?:sống|ở|sinh sống)\s+(?:tại|ở)\s+([A-Za-zÀ-ỹ\s]+?)(?:\s+(?:và|,|\.|$))", 0.85),
        _PatternEntry(r"(?:mình|tôi)\s+ở\s+([A-Za-zÀ-ỹ]+)", 0.75),
        _PatternEntry(r"đang\s+làm\s+việc\s+(?:ở|tại)\s+([A-Za-zÀ-ỹ]+)", 0.60),
    ]),
    ("profession", [
        _PatternEntry(r"(?:giờ|hiện tại|bây giờ|đang)\s+(?:chuyển sang|làm)\s+([A-Za-zÀ-ỹ\s]+?engineer[A-Za-zÀ-ỹ\s]*?)(?:\s+(?:cho|ở|tại|,|\.|$))", 0.90),
        _PatternEntry(r"(?:mình|tôi)\s+(?:đang\s+)?làm\s+([A-Za-zÀ-ỹ\s]+?engineer[A-Za-zÀ-ỹ\s]*?)(?:\s+(?:cho|ở|tại|,|\.|$))", 0.85),
        _PatternEntry(r"(?:mình|tôi)\s+(?:là\s+|làm\s+)([A-Za-zÀ-ỹ\s]+?(?:engineer|developer|manager|designer|analyst)[A-Za-zÀ-ỹ\s]*?)(?:\s+(?:cho|ở|tại|,|\.|$))", 0.80),
        _PatternEntry(r"nghề\s+(?:nghiệp\s+)?(?:mình|tôi|hiện tại)\s+(?:là\s+)?([A-Za-zÀ-ỹ\s]+?)(?:\s*[,\.]|$)", 0.75),
    ]),
    ("drink", [
        _PatternEntry(r"(?:đồ uống|thức uống)\s+(?:yêu thích|mình thích)\s+(?:là\s+)?([A-Za-zÀ-ỹ\s]+?)(?:\s*[,\.]|$)", 0.90),
        _PatternEntry(r"(?:mình|tôi)\s+(?:thích|uống)\s+([A-Za-zÀ-ỹ\s]+?(?:cà phê|trà|nước)[A-Za-zÀ-ỹ\s]*?)(?:\s*[,\.]|$)", 0.75),
        _PatternEntry(r"([A-Za-zÀ-ỹ\s]*cà phê[A-Za-zÀ-ỹ\s]*)\s+(?:là\s+)?(?:đồ uống|thức uống)\s+yêu thích", 0.85),
    ]),
    ("food", [
        _PatternEntry(r"(?:món ăn|đồ ăn)\s+yêu thích\s+(?:là\s+)?([A-Za-zÀ-ỹ\s]+?)(?:\s*[,\.]|$)", 0.90),
        _PatternEntry(r"(?:mình|tôi)\s+(?:thích ăn|ăn)\s+([A-Za-zÀ-ỹ\s]+?)(?:\s+(?:và|,|\.|$))", 0.70),
        _PatternEntry(r"(?:món ruột|món quen)\s+(?:của\s+(?:mình|tôi)\s+)?(?:là\s+)?([A-Za-zÀ-ỹ\s]+?)(?:\s*[,\.]|$)", 0.85),
    ]),
    ("pet", [
        _PatternEntry(r"(?:mình|tôi)\s+nuôi\s+(?:một?\s+(?:bé|con)\s+)?([A-Za-zÀ-ỹ\s]+?)(?:\s+tên|\s+(?:và|,|\.|$))", 0.85),
        _PatternEntry(r"(?:con|bé)\s+([A-Za-zÀ-ỹ]+)\s+(?:tên|của mình)", 0.80),
    ]),
    ("response_style", [
        _PatternEntry(r"(?:mình|tôi)\s+muốn\s+(?:bạn\s+)?trả lời\s+([A-Za-zÀ-ỹ\s,]+?)(?:\s*[,\.]|$)", 0.85),
        _PatternEntry(r"(?:trả lời|câu trả lời)\s+(?:ngắn gọn|súc tích|chi tiết|có bullet|thành bullet)([A-Za-zÀ-ỹ\s,]*?)(?:\s*[,\.]|$)", 0.80),
        _PatternEntry(r"(?:hãy|cần|nên)\s+trả lời\s+([A-Za-zÀ-ỹ\s,]+?)(?:\s*(?:khi|nếu|và|,|\.|$))", 0.70),
    ]),
]

# Keywords indicating the message is just a question (skip fact extraction)
_QUESTION_INDICATORS = [
    r"^(?:bạn\s+)?(?:có biết|có nhớ|nhớ|biết)\s",
    r"\?$",
    r"^(?:hỏi|thử hỏi)",
    r"^(?:tại sao|vì sao|như thế nào|làm sao|khi nào|ở đâu|ai|cái gì|điều gì)\b",
]

# Noise phrases — if present in the message, lower confidence by 0.2
# (e.g. "chỉ là câu đùa", "không phải", "ví dụ cũ")
_NOISE_PHRASES = [
    r"chỉ là câu đùa",
    r"chỉ là ví dụ",
    r"không phải",
    r"ví dụ cũ",
    r"đùa thôi",
    r"chứ không phải nơi ở",
]


def _is_question_only(message: str) -> bool:
    msg = message.strip().lower()
    for pat in _QUESTION_INDICATORS:
        if re.search(pat, msg):
            return True
    return False


def _noise_penalty(message: str) -> float:
    """Return confidence penalty (0.0–0.2) if noise phrases are detected."""
    lower = message.lower()
    for pat in _NOISE_PHRASES:
        if re.search(pat, lower):
            return 0.2
    return 0.0


def extract_profile_updates(
    message: str,
    min_confidence: float = MIN_CONFIDENCE_THRESHOLD,
) -> dict[str, str]:
    """Extract stable profile facts from a user message.

    BONUS — Confidence threshold:
        Each pattern carries a confidence score. Facts are only returned
        when their score meets `min_confidence`. Noise phrases detected in
        the message reduce the effective confidence by 0.2, preventing
        jokes or dismissive statements from being stored as facts.

    Returns a dict of {fact_key: value}.
    """
    if _is_question_only(message):
        return {}

    penalty = _noise_penalty(message)
    facts: dict[str, str] = {}

    for key, entries in _FACT_PATTERNS:
        for entry in entries:
            m = re.search(entry.pattern, message, re.IGNORECASE)
            if m:
                effective_confidence = entry.confidence - penalty
                if effective_confidence < min_confidence:
                    # Pattern matched but confidence too low — skip
                    break
                value = m.group(1).strip().rstrip(".,;")
                if len(value) >= 2:
                    facts[key] = value
                    break  # first confident match wins for this key

    return facts


# ---------------------------------------------------------------------------
# BONUS D — Structured entity extraction
# ---------------------------------------------------------------------------
# extract_profile_updates() returns a flat {key: value} dict — good enough
# for simple recall but loses structure inside each field.
#
# extract_entities() goes one level deeper:
#   - profession  → {role, domain}            e.g. "MLOps engineer" → role=MLOps, domain=engineer
#   - location    → {city, region}            e.g. "Đà Nẵng" → city=Đà Nẵng
#   - name        → {display_name, nickname}  e.g. "DũngCT" → display_name=DũngCT
#   - drink       → {item, modifier}          e.g. "cà phê sữa đá" → item=cà phê, modifier=sữa đá
#   - pet         → {species, pet_name}       extracted from "nuôi một bé corgi tên Bơ"
#
# Benefits:
#   - More precise matching when user asks "role" vs "domain"
#   - Easier to update one sub-field without clobbering others
#   - Facts stored as structured markdown sections
#
# Risk introduced:
#   - Sub-field extraction may fail for non-standard phrasing → falls back to raw value
#   - More complex User.md format may confuse simple line-based fact reader

class EntityBundle(TypedDict, total=False):
    """Structured representation of one extracted entity."""
    raw: str           # original extracted string (always present)
    role: str          # for profession: the role part  e.g. "MLOps"
    domain: str        # for profession: the domain     e.g. "engineer"
    city: str          # for location
    region: str        # for location (optional)
    display_name: str  # for name
    nickname: str      # for name (optional)
    item: str          # for drink/food
    modifier: str      # for drink: modifier e.g. "sữa đá"
    species: str       # for pet
    pet_name: str      # for pet


def _parse_profession(raw: str) -> EntityBundle:
    """Split 'MLOps engineer' → role='MLOps', domain='engineer'."""
    bundle: EntityBundle = {"raw": raw}
    roles = ["MLOps", "Backend", "Frontend", "Data", "DevOps", "Platform",
             "AI", "ML", "Software", "Full-stack", "Cloud", "Site Reliability"]
    domains = ["engineer", "developer", "manager", "designer", "analyst", "scientist"]
    raw_lower = raw.lower()
    for r in roles:
        if r.lower() in raw_lower:
            bundle["role"] = r
            break
    for d in domains:
        if d in raw_lower:
            bundle["domain"] = d
            break
    return bundle


def _parse_location(raw: str) -> EntityBundle:
    """Extract city from location string."""
    bundle: EntityBundle = {"raw": raw}
    # Common Vietnamese city names
    cities = ["Hà Nội", "TP HCM", "Hồ Chí Minh", "Đà Nẵng", "Huế", "Hội An",
              "Cần Thơ", "Nha Trang", "Đà Lạt", "Hải Phòng", "Vinh", "Buôn Ma Thuột"]
    for city in cities:
        if city.lower() in raw.lower():
            bundle["city"] = city
            break
    if "city" not in bundle:
        bundle["city"] = raw.strip()
    return bundle


def _parse_name(raw: str) -> EntityBundle:
    """Extract display name and optional nickname."""
    bundle: EntityBundle = {"raw": raw, "display_name": raw.strip()}
    # Nickname pattern: name in parentheses or after "hay còn gọi là"
    nick_m = re.search(r"\(([^)]+)\)", raw)
    if nick_m:
        bundle["nickname"] = nick_m.group(1).strip()
        bundle["display_name"] = raw[:nick_m.start()].strip()
    return bundle


def _parse_drink(raw: str) -> EntityBundle:
    """Split 'cà phê sữa đá' → item='cà phê', modifier='sữa đá'."""
    bundle: EntityBundle = {"raw": raw}
    base_drinks = ["cà phê", "trà", "nước", "sinh tố", "nước ép", "bia", "rượu"]
    for base in base_drinks:
        if base in raw.lower():
            bundle["item"] = base
            after = raw.lower().replace(base, "", 1).strip()
            if after:
                bundle["modifier"] = after.strip("- ")
            break
    if "item" not in bundle:
        bundle["item"] = raw.strip()
    return bundle


def _parse_pet(raw: str, original_message: str) -> EntityBundle:
    """Extract species and pet name."""
    bundle: EntityBundle = {"raw": raw, "species": raw.strip()}
    # Try to extract pet name from "tên X"
    name_m = re.search(r"tên\s+([A-Za-zÀ-ỹ]+)", original_message, re.IGNORECASE)
    if name_m:
        bundle["pet_name"] = name_m.group(1).strip()
    return bundle


def extract_entities(message: str, min_confidence: float = MIN_CONFIDENCE_THRESHOLD) -> dict[str, EntityBundle]:
    """BONUS D — Structured entity extraction.

    Returns a dict of {fact_key: EntityBundle} where each bundle contains
    the raw extracted value PLUS structured sub-fields when parseable.

    Example:
        "Mình làm MLOps engineer" →
        {"profession": {"raw": "MLOps engineer", "role": "MLOps", "domain": "engineer"}}

    Falls back gracefully: if sub-field parsing fails, bundle still has "raw".
    Uses the same confidence threshold as extract_profile_updates().
    """
    flat = extract_profile_updates(message, min_confidence=min_confidence)
    result: dict[str, EntityBundle] = {}

    for key, raw_val in flat.items():
        if key == "profession":
            result[key] = _parse_profession(raw_val)
        elif key == "location":
            result[key] = _parse_location(raw_val)
        elif key == "name":
            result[key] = _parse_name(raw_val)
        elif key == "drink":
            result[key] = _parse_drink(raw_val)
        elif key == "pet":
            result[key] = _parse_pet(raw_val, message)
        else:
            result[key] = EntityBundle(raw=raw_val)

    return result


# ---------------------------------------------------------------------------
# BONUS B — Memory decay
# ---------------------------------------------------------------------------
# FactRecord stores a fact alongside metadata used for decay:
#   - value      : the current string value
#   - written_at : unix timestamp when last written
#   - mentions   : how many times this fact has been confirmed / repeated
#
# Decay logic in `decayed_facts()`:
#   - Facts with zero or few mentions decay first.
#   - A fact is considered "stale" after DECAY_STALE_SECONDS seconds without
#     being re-mentioned, unless it has been mentioned >= DECAY_MENTION_FLOOR times.
#   - Stale facts are returned with a "~" prefix so callers can decide whether
#     to trust them (soft decay — we never hard-delete from User.md automatically).
#
# Risk introduced:
#   - A very stable fact (e.g. name) that the user never repeats could be
#     marked stale over time.  The DECAY_MENTION_FLOOR constant mitigates this.
#   - Decay metadata lives in memory only; restarting the process resets it.
#     A production system should persist FactRecord to disk.

DECAY_STALE_SECONDS: float = 3600.0   # 1 hour in production; lower in tests
DECAY_MENTION_FLOOR: int = 3           # default floor — overridden per key below

# FIX 3 — Per-key decay floor:
# Stable identity facts (name) should never decay.
# Contextual facts (location, profession) decay after fewer confirmations.
# Preference facts (drink, food, style) decay at the default rate.
DECAY_FLOOR_PER_KEY: dict[str, int] = {
    "name":           999,   # never decay — identity is permanent
    "pet":            999,   # pet name is stable
    "location":         2,   # may change; decay after 2 mentions if not refreshed
    "profession":       2,   # may change with job changes
    "drink":            3,   # default
    "food":             3,   # default
    "response_style":   1,   # style preference — low bar, any mention counts
}


@dataclass
class FactRecord:
    value: str
    written_at: float = field(default_factory=time.time)
    mentions: int = 1


@dataclass
class MemoryDecayStore:
    """Decay-aware wrapper around UserProfileStore.

    BONUS B — Memory decay with three improvements over v1:

    FIX 1 — Per-key decay floor (DECAY_FLOOR_PER_KEY):
        'name' and 'pet' never decay (floor=999).
        'location' and 'profession' decay sooner (floor=2) since they change with life events.
        Other facts use their individual floor from the config dict.

    FIX 2 — Persist decay metadata to disk (decay_state.json):
        FactRecord data is serialised to a JSON sidecar file next to User.md.
        On init, if the file exists it is loaded back, so decay state survives
        process restarts. Without this, every restart resets all mention counts
        and the decay system behaves as if every fact is brand new.

    Risk introduced by persistence:
        The sidecar file can become stale if User.md is edited externally.
        A startup consistency check (see _load_records) drops records whose
        stored value no longer matches User.md to prevent ghost entries.
    """

    profile_store: UserProfileStore
    stale_seconds: float = DECAY_STALE_SECONDS
    mention_floor: int = DECAY_MENTION_FLOOR          # fallback if key not in per-key map
    _records: dict[str, dict[str, FactRecord]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Eagerly restore persisted decay state for all users found on disk
        for md_path in self.profile_store.root_dir.glob("*.md"):
            user_id = md_path.stem
            self._load_records(user_id)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _sidecar_path(self, user_id: str) -> Path:
        """Return path to the JSON sidecar file for decay metadata."""
        base = self.profile_store.path_for(user_id)
        return base.with_suffix(".decay.json")

    def _load_records(self, user_id: str) -> None:
        """Load persisted FactRecords from disk, pruning stale entries."""
        import json
        sidecar = self._sidecar_path(user_id)
        if not sidecar.exists():
            return
        try:
            raw = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:
            return  # corrupt file — start fresh

        current_facts = self.profile_store.facts(user_id)
        records: dict[str, FactRecord] = {}
        for key, rec in raw.items():
            stored_value = rec.get("value", "")
            # Consistency check: drop if value no longer matches User.md
            if current_facts.get(key) != stored_value:
                continue
            records[key] = FactRecord(
                value=stored_value,
                written_at=float(rec.get("written_at", time.time())),
                mentions=int(rec.get("mentions", 1)),
            )
        self._records[user_id] = records

    def _save_records(self, user_id: str) -> None:
        """Persist FactRecords for a user to a JSON sidecar file."""
        import json
        records = self._records.get(user_id, {})
        sidecar = self._sidecar_path(user_id)
        data = {
            key: {
                "value": rec.value,
                "written_at": rec.written_at,
                "mentions": rec.mentions,
            }
            for key, rec in records.items()
        }
        sidecar.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _floor_for(self, key: str) -> int:
        """Return the per-key mention floor, falling back to global default."""
        return DECAY_FLOOR_PER_KEY.get(key, self.mention_floor)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        """Write fact to User.md and update decay metadata."""
        if user_id not in self._records:
            self._load_records(user_id)
        records = self._records.setdefault(user_id, {})
        existing = records.get(key)

        if existing and existing.value == value:
            existing.mentions += 1
            existing.written_at = time.time()
        else:
            records[key] = FactRecord(value=value)

        self.profile_store.upsert_fact(user_id, key, value)
        self._save_records(user_id)

    def touch_fact(self, user_id: str, key: str) -> None:
        """Increment mention count for a fact without changing its value."""
        if user_id not in self._records:
            self._load_records(user_id)
        records = self._records.get(user_id, {})
        if key in records:
            records[key].mentions += 1
            records[key].written_at = time.time()
            self._save_records(user_id)

    def decayed_facts(self, user_id: str) -> dict[str, str]:
        """Return facts dict, marking stale entries with a '~' prefix.

        Uses per-key floor (DECAY_FLOOR_PER_KEY) so identity facts like
        'name' are never marked stale while contextual facts like 'location'
        decay sooner.
        """
        if user_id not in self._records:
            self._load_records(user_id)
        raw_facts = self.profile_store.facts(user_id)
        records = self._records.get(user_id, {})
        now = time.time()
        result: dict[str, str] = {}

        for key, value in raw_facts.items():
            floor = self._floor_for(key)
            # Special case: floor >= 999 means "never decay"
            if floor >= 999:
                result[key] = value
                continue
            record = records.get(key)
            if record is None:
                result[key] = value
                continue
            age = now - record.written_at
            if record.mentions < floor and age > self.stale_seconds:
                result[key] = f"~{value}"
            else:
                result[key] = value

        return result

    # Proxy methods
    def read_text(self, user_id: str) -> str:
        return self.profile_store.read_text(user_id)

    def write_text(self, user_id: str, content: str) -> Path:
        return self.profile_store.write_text(user_id, content)

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        return self.profile_store.edit_text(user_id, search_text, replacement)

    def file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def facts(self, user_id: str) -> dict[str, str]:
        return self.profile_store.facts(user_id)

    def path_for(self, user_id: str) -> Path:
        return self.profile_store.path_for(user_id)


# ---------------------------------------------------------------------------
# Message summarizer
# ---------------------------------------------------------------------------

def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact text summary of a list of messages.

    Takes the most recent `max_items` messages and concatenates them
    as a readable summary. This is a heuristic approach — no LLM needed.
    """
    if not messages:
        return ""

    recent = messages[-max_items:] if len(messages) > max_items else messages
    lines: list[str] = []
    for msg in recent:
        role = msg.get("role", "unknown")
        content = msg.get("content", "").strip()
        if not content:
            continue
        # Truncate very long lines to keep summary compact
        if len(content) > 200:
            content = content[:197] + "..."
        prefix = "User" if role == "user" else "Assistant"
        lines.append(f"{prefix}: {content}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CompactMemoryManager
# ---------------------------------------------------------------------------

@dataclass
class CompactMemoryManager:
    """Compact memory manager for long conversation threads.

    Keeps recent messages in full. When the thread grows beyond
    `threshold_tokens`, older messages are summarized and discarded,
    keeping only the most recent `keep_messages` turns.
    """

    threshold_tokens: int
    keep_messages: int
    # FIX 1 — Summary length cap:
    # Without a cap, the summary grows unboundedly as compactions stack up,
    # eventually costing more tokens than the original messages.
    # We cap the summary at max_summary_tokens (default 400 ≈ 1,600 chars).
    # When the cap is hit, we keep only the tail of the existing summary
    # (most recent history) before appending new content.
    max_summary_tokens: int = 400
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _init_thread(self, thread_id: str) -> None:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }

    def append(self, thread_id: str, role: str, content: str) -> None:
        """Append a message and trigger compaction if token budget is exceeded."""
        self._init_thread(thread_id)
        thread = self.state[thread_id]
        thread["messages"].append({"role": role, "content": content})  # type: ignore[index]

        # Check if we need to compact
        total_tokens = self._count_tokens(thread_id)
        if total_tokens > self.threshold_tokens:
            self._compact(thread_id)

    def _count_tokens(self, thread_id: str) -> int:
        thread = self.state[thread_id]
        msgs: list[dict[str, str]] = thread["messages"]  # type: ignore[assignment]
        summary: str = thread["summary"]  # type: ignore[assignment]
        total = estimate_tokens(summary)
        for msg in msgs:
            total += estimate_tokens(msg.get("content", ""))
        return total

    def _compact(self, thread_id: str) -> None:
        """Move older messages into a summary, keep only the most recent ones.

        FIX 1 — Summary length cap:
            After building the new summary, if it exceeds max_summary_tokens
            we truncate the HEAD (oldest part) and keep the TAIL (most recent),
            prefixing with '[...earlier history truncated...]'.
            This prevents the summary from growing O(n) with each compaction,
            bounding prompt overhead to O(max_summary_tokens + keep_messages).
        """
        thread = self.state[thread_id]
        msgs: list[dict[str, str]] = thread["messages"]  # type: ignore[assignment]
        existing_summary: str = thread["summary"]  # type: ignore[assignment]

        if len(msgs) <= self.keep_messages:
            return

        old_msgs = msgs[: -self.keep_messages]
        kept_msgs = msgs[-self.keep_messages :]

        old_summary_text = summarize_messages(old_msgs, max_items=len(old_msgs))
        if existing_summary:
            new_summary = existing_summary.rstrip() + "\n\n[Earlier summary continued]\n" + old_summary_text
        else:
            new_summary = "[Conversation summary]\n" + old_summary_text

        # Apply summary length cap — truncate head if over budget
        if estimate_tokens(new_summary) > self.max_summary_tokens:
            # Keep approximately the last max_summary_tokens worth of chars
            cap_chars = self.max_summary_tokens * 4
            if len(new_summary) > cap_chars:
                new_summary = "[...earlier history truncated...]\n" + new_summary[-cap_chars:]

        thread["messages"] = kept_msgs
        thread["summary"] = new_summary
        thread["compactions"] = int(thread["compactions"]) + 1  # type: ignore[arg-type]

    def context(self, thread_id: str) -> dict[str, object]:
        """Return per-thread state with keys: messages, summary, compactions."""
        self._init_thread(thread_id)
        return dict(self.state[thread_id])

    def compaction_count(self, thread_id: str) -> int:
        """Return number of compactions performed for this thread."""
        self._init_thread(thread_id)
        return int(self.state[thread_id]["compactions"])  # type: ignore[arg-type]
