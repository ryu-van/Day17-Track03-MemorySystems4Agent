# Phân tích kết quả benchmark — Day 17 Track 3: Memory Systems for AI Agent

> Benchmark chạy trên hai chế độ:
> - **Offline** (`force_offline=True`): heuristic deterministic, không gọi API, dùng để kiểm tra cấu trúc memory.
> - **Live** (`force_offline=False`): gọi LLM thật qua AntcoAI Gateway (model `gemini-3.1-flash-lite`), phản ánh chất lượng thực tế.

---

## Kết quả benchmark

### Standard Benchmark — Offline (`data/conversations.json`, 10 phiên)

| Agent    | Agent tokens | Prompt tokens | Cross-session recall | Response quality | Memory growth | Compactions |
|----------|-------------|---------------|----------------------|-----------------|---------------|-------------|
| Baseline | 2,915       | 20,054        | 0.000                | 0.100           | 0 bytes       | 0           |
| Advanced | 2,858       | 26,929        | **0.429**            | **0.506**       | 274 bytes     | 0           |

### Standard Benchmark — Live (LLM thật, v3 — sau fix)

| Agent           | Agent tokens | Prompt tokens | Cross-session recall | Response quality | Memory growth | Compactions |
|-----------------|-------------|---------------|----------------------|-----------------|---------------|-------------|
| Baseline (live) | 31,822      | 1,625         | 0.107                | 0.207           | 0 bytes       | 0           |
| Advanced (live) | 16,817      | **44,536**    | **0.643**            | **0.707**       | 274 bytes     | 43          |

> Trước fix: Advanced prompt = 210,194. Sau fix: **44,536 (-79%)**.

### Long-Context Stress Benchmark — Offline (`data/advanced_long_context.json`, 1 phiên dài 16 lượt)

| Agent    | Agent tokens | Prompt tokens | Cross-session recall | Response quality | Memory growth | Compactions |
|----------|-------------|---------------|----------------------|-----------------|---------------|-------------|
| Baseline | 407         | 23,273        | 0.000                | 0.100           | 0 bytes       | 0           |
| Advanced | 310         | **15,501**    | 0.167                | 0.253           | 129 bytes     | **16**      |

### Long-Context Stress Benchmark — Live (LLM thật, v3 — sau fix)

| Agent           | Agent tokens | Prompt tokens | Cross-session recall | Response quality | Memory growth | Compactions |
|-----------------|-------------|---------------|----------------------|-----------------|---------------|-------------|
| Baseline (live) | 4,011       | 2,376         | 0.000                | 0.100           | 0 bytes       | 0           |
| Advanced (live) | 5,038       | **13,089**    | **0.500**            | **0.567**       | 129 bytes     | **26**      |

> Trước fix: Advanced stress prompt = 71,794. Sau fix: **13,089 (-82%)**. Prompt ổn định ~875 tokens/turn thay vì tăng tuyến tính.

### Bonus Benchmark — Confidence Threshold & Entity Extraction (117 turns)

| Metric                                      | Giá trị                        |
|---------------------------------------------|--------------------------------|
| Facts extracted WITHOUT threshold           | 21                             |
| Facts extracted WITH threshold              | 19                             |
| Facts bị block bởi threshold               | **2 (9.5% filtered out)**      |
| Noise turns được phát hiện                  | 7                              |
| Noise turns bị block đúng                  | 1 (14.3% of noise turns)       |
| Structured sub-fields extracted             | **17 sub-fields**              |

---

## 1. Tại sao Advanced có recall tốt hơn Baseline?

**Offline**: Baseline recall = 0.000, Advanced = 0.429 (+42.9pp).
**Live**: Baseline recall = 0.107, Advanced = **0.679** (+57.2pp).

Baseline chỉ giữ messages trong cùng `thread_id`. Khi recall question hỏi trong thread mới, context bị xóa hoàn toàn. Baseline live có recall 0.107 (khác offline = 0) vì LLM thỉnh thoảng đoán được tên từ pattern, không phải từ memory thật.

Advanced lưu facts vào `User.md` trên disk sau mỗi lượt. Khi sang thread mới, agent đọc lại `User.md` và inject vào system prompt → LLM có context đầy đủ để trả lời. Live mode cho thấy recall tăng từ 0.429 (offline heuristic) lên **0.679** (LLM thật hiểu ngữ nghĩa tốt hơn).

**Câu chuyện trong data**: conv-03 (Đà Nẵng → Huế) và conv-06 (backend → MLOps) được xử lý đúng nhờ `upsert_fact()` ghi đè fact cũ và conflict handling log correction.

---

## 2. Tại sao Advanced tốn prompt tokens hơn Baseline?

### Offline
Advanced: 26,929 vs Baseline: 20,054 (+34%).

### Live — gap lớn hơn nhiều
Advanced: **210,194** vs Baseline: **1,625** — chênh lệch 129x.

Lý do ở live mode cực đoan hơn:
- **Baseline live**: LangGraph với `InMemorySaver` chỉ track messages ngắn, không kéo lịch sử dài. Prompt tokens nhỏ vì mỗi turn chỉ có system prompt + message hiện tại.
- **Advanced live**: Mỗi turn inject toàn bộ User.md + compact summary + recent messages vào system prompt, cộng thêm LangGraph kéo lại lịch sử tool calls. 54 compactions trong 10 phiên chứng tỏ hội thoại vượt ngưỡng liên tục.

**Kết luận**: User.md overhead ở live mode lớn hơn dự kiến vì LLM cần full context để trả lời chính xác. Đây là trade-off rõ ràng: recall tốt hơn nhưng prompt cost cao hơn đáng kể.

---

## 3. Tại sao compact giúp Advanced ở hội thoại dài?

### Offline (stress)
Baseline: 23,273 tokens vs Advanced: 15,501 tokens → Advanced tiết kiệm **-33.3%** prompt tokens.

### Live (stress)
Baseline: 2,376 tokens vs Advanced: 71,794 tokens.

Live stress cho kết quả ngược với offline — Advanced tốn hơn Baseline. Lý do:

- **Baseline live ở stress**: chỉ có 1 phiên dài, LangGraph baseline không kéo lịch sử quá dài → prompt nhỏ.
- **Advanced live ở stress**: 26 compactions → summary ngày càng dài và được kéo vào mọi turn. Mỗi turn cộng: User.md + summary dài + recent messages + tool call history.

**Điều này tiết lộ giới hạn của compact heuristic**: summary dựa trên text concatenation không thật sự nén — nó chỉ cắt tin nhắn cũ nhưng summary text vẫn tăng theo thời gian. Compact chỉ thực sự thắng khi summary ngắn hơn tổng lịch sử bị thay thế — điều này cần LLM-based summarization để đạt được trong production.

**Kết luận quan trọng**: Compact chủ yếu tối ưu `prompt tokens processed` trong **offline mode**. Ở live mode với LLM thật, compact cần LLM summarizer thực sự (không phải text concatenation) mới có hiệu quả rõ rệt.

---

## 4. Bonus A — Confidence Threshold

**Vấn đề giải quyết**: Agent ghi bất kỳ regex match nào vào User.md, kể cả khi user nói đùa ("hay là chuyển sang product manager") hay nhắc thông tin cũ.

**Cách hoạt động**:
- Mỗi pattern có confidence 0.60–0.95.
- Noise phrases ("đùa thôi", "chỉ là câu đùa", "chứ không phải nơi ở") trừ 0.20 penalty.
- Chỉ facts với `effective_confidence ≥ 0.65` được ghi.

**Số liệu**: 9.5% facts bị filter (2/21), 14.3% noise turns bị block đúng (1/7).

**Cải thiện recall**: User.md sạch hơn → LLM không bị confused bởi facts sai → recall chính xác hơn. Live benchmark cho thấy Advanced recall 0.679 — phần này đến từ profile sạch.

**Rủi ro**: Facts nói gián tiếp (confidence 0.60) không bao giờ lưu được. User cần phát biểu rõ ràng. Per-key threshold sẽ tốt hơn global constant.

---

## 5. Bonus B — Memory Decay

**Vấn đề giải quyết**: Facts cũ chưa được xác nhận lại tích lũy trong User.md mãi, agent tin tuyệt đối dù có thể lỗi thời.

**Cách hoạt động**:
- `MemoryDecayStore` track `written_at` và `mentions` cho mỗi fact.
- Fact *stale* khi: `mentions < 3` VÀ `age > 3600s`.
- Stale facts trả về với prefix `~` (soft decay, không xóa).
- `touch_fact()` reset decay clock khi user xác nhận lại.

**Cải thiện recall**: Agent ưu tiên facts tươi hơn, có thể trả lời "mình không chắc về thông tin này" thay vì trả lời sai tự tin.

**Rủi ro**: `name` nếu không nhắc lại 3 lần trong 1 giờ sẽ bị đánh dấu stale — không hợp lý. Cần per-key floor: `name` không decay, `location` decay sau 7 ngày.

---

## 6. Bonus C — Conflict Handling

**Vấn đề giải quyết**: Khi user đính chính, cần đảm bảo fact mới ghi đè hoàn toàn, không giữ fact cũ sai song song.

**Cách hoạt động**:
- `upsert_fact()` dùng regex replace → không thể có 2 dòng cùng key.
- `_reply_offline()` inject `[Correction detected: 'key' X → Y]` vào compact memory khi phát hiện thay đổi.

**Cải thiện**: conv-06 (backend → MLOps) và stress-01 (Huế → Đà Nẵng) đều được xử lý đúng. Recall cho recall questions về nghề nghiệp và nơi ở tăng rõ.

**Rủi ro**: Nếu user nói sai rồi sửa lại ngay lập tức, cả 2 changes đều được ghi vào compact memory, tạo noise. Cần confidence threshold để chỉ trigger correction khi có đủ chắc chắn.

---

## 7. Bonus D — Structured Entity Extraction

**Vấn đề giải quyết**: Flat string "MLOps engineer" mất cấu trúc bên trong — không phân biệt role vs domain.

**Cách hoạt động** (`extract_entities()`):
- `profession` → `{raw, role, domain}`: "MLOps engineer" → role=MLOps, domain=engineer
- `location` → `{raw, city}`: "Đà Nẵng" → city=Đà Nẵng
- `name` → `{raw, display_name, nickname}`
- `drink` → `{raw, item, modifier}`: "cà phê sữa đá" → item=cà phê, modifier=sữa đá
- `pet` → `{raw, species, pet_name}`: "corgi tên Bơ" → species=corgi, pet_name=Bơ

**Số liệu**: 17 structured sub-fields từ 19 facts (0.9 sub-fields/fact).

**Cải thiện**: Recall chính xác hơn cho câu hỏi dạng "role hiện tại là gì?" vs "làm lĩnh vực gì?". Update một sub-field không cần ghi đè toàn bộ fact.

**Rủi ro**: Danh sách cứng (cities, roles, domains) không nhận ra entities mới. Cần NER model hoặc LLM extraction cho production. Fallback về `raw` đảm bảo không mất data.

---

## 10. Ba fix hoàn thiện hệ thống (production-ready)

### Fix 1 — Summary Length Cap (`CompactMemoryManager.max_summary_tokens=400`)

**Vấn đề**: Compact heuristic cộng dồn summary mỗi lần compact → sau nhiều compactions, summary tự nó to hơn lịch sử ban đầu. Live stress test cho thấy Advanced tốn 71,794 prompt tokens dù compact 26 lần — vì summary phình không giới hạn.

**Cách fix**: `_compact()` sau khi build summary mới kiểm tra `estimate_tokens(summary) > max_summary_tokens`. Nếu vượt, cắt đầu summary (phần cũ nhất) và giữ đuôi, prefix `[...earlier history truncated...]`. Kết quả: summary bị giới hạn ở `max_summary_tokens × 4 chars` — không tăng O(n).

**Test**: `test_summary_length_cap` gửi 40 messages, xác nhận summary tokens ≤ 120 dù có nhiều compactions.

**Cải thiện token cost**: Prompt overhead của Advanced ở live stress sẽ giảm đáng kể trong production — summary không còn phình vô hạn.

**Rủi ro mới**: Truncation mất thông tin đầu summary (history cũ nhất). Acceptable vì thông tin quan trọng đã được extract vào User.md — compact chỉ giữ context ngắn hạn.

---

### Fix 2 — Persist Decay Metadata (`MemoryDecayStore` → `.decay.json` sidecar)

**Vấn đề**: Decay metadata (`written_at`, `mentions`) chỉ sống trong RAM. Restart process → mất toàn bộ → mọi fact bị coi là mới 1 mention → decay không hoạt động đúng sau restart.

**Cách fix**: Mỗi lần `upsert_fact()` hoặc `touch_fact()` → ghi `{key: {value, written_at, mentions}}` ra file `<user_id>.decay.json` cạnh `<user_id>.md`. Khi `MemoryDecayStore.__post_init__()` chạy, nó scan `*.md` files và load sidecar tương ứng. Có consistency check: nếu value trong sidecar khác User.md (User.md bị edit ngoài) → bỏ record đó.

**Test**: `test_decay_persists_across_restarts` tạo store1, upsert + touch 3 lần, tạo store2 mới cùng dir → xác nhận facts không bị stale.

**Cải thiện**: Decay system hoạt động đúng trong production multi-session thay vì chỉ trong một process.

**Rủi ro mới**: Sidecar file thêm I/O overhead mỗi lần upsert/touch. Với user profile nhỏ (<50 facts) overhead không đáng kể. Nếu User.md bị xóa nhưng sidecar còn → sidecar trở nên orphan (harmless, không load được do không có User.md để cross-check).

---

### Fix 3 — Per-Key Decay Floor (`DECAY_FLOOR_PER_KEY`)

**Vấn đề**: Global `DECAY_MENTION_FLOOR=3` áp dụng cho tất cả facts — không hợp lý vì `name` là identity fact không bao giờ nên decay, nhưng `location` có thể thay đổi theo tháng.

**Cách fix**:
```python
DECAY_FLOOR_PER_KEY = {
    "name":           999,   # never decay
    "pet":            999,   # stable identity
    "location":         2,   # may change with moves
    "profession":       2,   # may change with jobs
    "drink":            3,   # default
    "food":             3,   # default
    "response_style":   1,   # low bar
}
```
`floor >= 999` → skip decay check hoàn toàn. `_floor_for(key)` tra cứu dict, fallback về `self.mention_floor`.

**Tests**:
- `test_per_key_decay_floor_name_never_decays`: name với 1 mention, stale_seconds=0 → không bao giờ stale.
- `test_per_key_decay_floor_location_decays_sooner`: cùng điều kiện, location (floor=2) với 1 mention → stale, name → không stale.

**Cải thiện recall**: Agent không bao giờ đánh dấu tên user là stale → recall câu hỏi về tên luôn chính xác. Location được decay sớm hơn → agent ít trả lời tự tin với địa chỉ cũ.

**Rủi ro mới**: `name floor=999` nghĩa là tên sai (user nhập nhầm) không bao giờ được decay. Cần conflict handling (đã có) để override khi user đính chính rõ ràng.

---

## 11. Tóm tắt toàn bộ hệ thống (final)

```
Tests: 18/18 passed
```

**Features implemented**:

| Feature | Mô tả | Giải quyết |
|---------|-------|-----------|
| Baseline Agent | within-session memory only | mốc so sánh |
| Advanced Agent | 3 lớp: short-term + User.md + compact | recall cross-session |
| Compact Memory | nén lịch sử khi > threshold | giảm prompt tokens |
| **Fix 1**: Summary Cap | giới hạn summary ≤ 400 tokens | compact không phình |
| **Bonus A**: Confidence threshold | filter facts theo score | giảm false positive 9.5% |
| **Bonus B**: Memory decay | stale marking + persist sidecar | facts cũ giảm ưu tiên |
| **Fix 2**: Persist decay | `.decay.json` sidecar | decay state survive restart |
| **Fix 3**: Per-key floor | `name=999`, `location=2`... | identity facts không decay |
| **Bonus C**: Conflict handling | correction ghi đè + log | không giữ fact sai song song |
| **Bonus D**: Entity extraction | structured sub-fields | recall chính xác hơn |

**Câu chuyện kín**:
1. Baseline → recall=0 cross-session
2. Advanced + User.md → recall=0.679 live (+57pp)
3. Hội thoại dài → prompt tăng O(n²) ở baseline
4. Compact + Summary Cap → giới hạn prompt overhead
5. Confidence + Decay + Per-key floor → User.md sạch, đúng, không phình
6. Persist + Conflict + Entity → production-ready, không mất state, structured recall

| Chế độ      | Benchmark   | Memory growth |
|-------------|-------------|---------------|
| Offline     | Standard    | 274 bytes     |
| Live        | Standard    | 274 bytes     |
| Offline     | Stress      | 129 bytes     |
| Live        | Stress      | 129 bytes     |

Memory growth nhất quán giữa offline và live vì cả hai đều dùng cùng `upsert_fact()` logic.

**Tốc độ tăng**: ~27 bytes/phiên (standard). Sau 1,000 phiên ≈ 27KB — nhỏ về storage nhưng là 27KB × N turns prompt overhead mỗi lượt.

**Rủi ro**: Không có upper bound. Cần periodic pruning hoặc hard limit (ví dụ max 50 facts, khi vượt → xóa facts stale nhất).

---

## 9. Tóm tắt trade-off toàn hệ thống

```
Metric              Baseline(off) Advanced(off) Baseline(live) Advanced(live,v3)
─────────────────── ──────────── ───────────── ────────────── ─────────────────
Agent tokens         2,915         2,858         31,822         16,817
Prompt tokens(std)  20,054        26,929          1,625         44,536  (v1:210k)
Prompt tokens(str)  23,273        15,501          2,376         13,089  (v1:71k)
Recall (std)         0.000         0.429          0.107          0.643
Quality (std)        0.100         0.506          0.207          0.707
Compactions (std)       0             0              0             43
Memory growth           0           274B             0            274B
```

**Câu chuyện kín**:
1. **Baseline không nhớ dài hạn** — recall 0.000 offline, 0.107 live (LLM đoán ngẫu nhiên).
2. **Advanced thêm User.md** — recall tăng lên 0.429 offline, **0.643 live** (+53pp vs baseline live).
3. **Hội thoại dài làm prompt cost tăng mạnh** — LangGraph InMemorySaver replay O(n²): v1 Advanced live stress = 71,794 tokens.
4. **Compact + Summary Cap + Compact-aware live path** — prompt ổn định ~875 tokens/turn ở stress. Advanced live stress v3 = **13,089 tokens (-82% so với v1)**.
5. **Hệ thống mạnh hơn nhưng cần guardrail tốt hơn** — confidence threshold (9.5% filter), memory decay + per-key floor (name không bao giờ stale), conflict handling (correction ghi đè đúng), entity extraction (17 sub-fields), persist decay sidecar (survive restart).

**Kết luận kỹ thuật**: Live mode xác nhận rằng persistent memory (User.md) là tính năng có giá trị thực — recall tăng từ 0.107 lên 0.679 (+57pp) chỉ nhờ thêm profile context. Compact memory heuristic hoạt động tốt ở offline nhưng cần nâng cấp thành LLM-based summarization để thực sự tối ưu prompt tokens ở production.
