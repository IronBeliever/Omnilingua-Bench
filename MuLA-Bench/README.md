---
license: cc-by-nc-4.0
task_categories:
  - audio-text-to-text
  - question-answering
language:
  - ar
  - de
  - en
  - es
  - fr
  - hi
  - id
  - it
  - ja
  - ko
  - pt
  - ru
  - th
  - tr
  - vi
  - zh
tags:
  - long-form-audio
  - multilingual
  - audio-understanding
  - omni
  - evaluation
  - question-answering
  - acoustic-events
pretty_name: MuLA-Bench
size_categories:
  - 1K<n<10K
---

# MuLA-Bench

**MuLA-Bench: A Multilingual Long-form Audio Understanding Benchmark via Multi-Tier Auditing**

MuLA-Bench is a **multilingual, long-form audio understanding** benchmark. Each
item is a question over a naturally occurring long recording; the model listens
to the source audio and answers from spoken content, natural acoustic events, or both.

- **5,038 questions** over **1,732 source recordings** (**1,341 hours**)
- **16 languages × 8 domains**, with **30 semantic questions per cell**
- Two evidence tracks: **semantic** (3,840 questions) and **acoustic** (1,198 questions)
- Four operations: **factual / reasoning / temporal / long-range**
- Multi-tier construction: evidence grounding, programmatic gates, shortcut checks, and language-expert review

## Why this benchmark

Long-context capacity and aggregate accuracy can conceal different kinds of
failure. In MuLA-Bench, **the same long recording** can support different
operations on different evidence types: a spoken fact, natural applause, a
clock-grounded event, or two cues tens of minutes apart. Overall accuracy on the
5,038-item freeze is therefore not a single skill score: two models can match on
the headline and still have different semantic / acoustic profiles.

The semantic track balances question counts and task composition across all
128 Language × Domain cells while retaining the original recordings. The acoustic
track follows verified natural event availability, without inserting sounds or
forcing equal coverage.

### Per-language inventory

Semantic coverage is **30 × 8 domains = 240 questions per language**. Each
semantic cell contains **8 factual, 8 reasoning, 7 temporal, and 7 long-range**
questions. Acoustic coverage is **not** forced to a flat 10 per cell; two cells,
`tr/howto` and `vi/lifestyle`, are empty.

**Table 1 — questions and distinct source recordings by language**

| Language | Code | Questions | Sources |
|----------|------|----------:|--------:|
| Arabic | ar | 313 | 106 |
| German | de | 304 | 118 |
| English | en | 347 | 123 |
| Spanish | es | 316 | 113 |
| French | fr | 338 | 98 |
| Hindi | hi | 290 | 124 |
| Indonesian | id | 324 | 127 |
| Italian | it | 304 | 87 |
| Japanese | ja | 320 | 101 |
| Korean | ko | 333 | 99 |
| Portuguese | pt | 306 | 106 |
| Russian | ru | 308 | 114 |
| Thai | th | 309 | 124 |
| Turkish | tr | 301 | 104 |
| Vietnamese | vi | 288 | 81 |
| Chinese | zh | 337 | 107 |
| **Total** | | **5,038** | **1,732** |

### Task × evidence

**Table 2 — official freeze by operation and evidence type**

| Operation (`task_type`) | Semantic | Acoustic | Total |
|-------------------------|---------:|---------:|------:|
| Factual / completeness (`completeness_fact`) | 1,024 | 867 | 1,891 |
| Reasoning (`reasoning`) | 1,024 | 122 | 1,146 |
| Temporal (`time_localization`) | 896 | 112 | 1,008 |
| Long-range (`long_range`) | 896 | 97 | 993 |
| **Total** | **3,840** | **1,198** | **5,038** |

Domains (**question counts**): lifestyle 672, knowledge 640, news 646, belief 635,
narrative 630, product 628, entertainment 596, howto 591. These are question-level
counts, distinct from the source-recording distribution in the paper's Figure 2.

Source duration is the original recording length: median **36.7 minutes**, mean
**46.5 minutes**, and **91.9%** of recordings last at least 30 minutes. These
statistics describe the 1,732 distinct source recordings, not question counts or
model-specific input lengths.

Evaluation uses the canonical audio defined under **Fair input** below. Four
predefined questions whose evidence extends beyond the three-hour mark use a
fixed later-stage three-hour segment. These are **four questions**, not four
source recordings; all four refer to source `Hgi8zNQnEa8`. Evidence timestamps
remain on the original source clock.

## Data layout

```text
mula-bench/
├── MuLA-Bench_v202609_public.jsonl
└── README.md                 # this file
```

The paper's official evaluation set contains **5,038 unique `id`s**. The statistics
in this README describe that frozen set. Before evaluating a downloaded
`MuLA-Bench_v202609_public.jsonl`, check that its inventory matches the paper using
the loading checks below.

### JSONL schema

| Field (JSON path) | Type | Description |
|-------------------|------|-------------|
| `id` | string | Stable item id, e.g. `nonac\|ar\|belief\|-aR600r70zQ\|80b787bc2ad8`. |
| `languages.source` | list | Source language code(s), e.g. `["ar"]`. `target` / `instruction` mirror it. |
| `input.messages` | list | Chat payload: an `audio` part (`media_id`) plus the `text` question in the source language. |
| `input.media` | list | Media entries with `id` / `type` / `path` (e.g. `media/audio/<video_id>.mp3`) / `duration_sec`. |
| `reference.answers` | list | Reference answer(s). **Not** shown to the model. |
| `reference.evidence_text` | string | Timestamped evidence used to write the item. **Not** shown to the model. |
| `metadata.video_id` | string | Social-media / source recording id. |
| `metadata.source_url` | string | Social-media URL of the source recording. |
| `metadata.topic` | string | Domain: `knowledge` / `howto` / `narrative` / `entertainment` / `lifestyle` / `news` / `product` / `belief`. |
| `metadata.cell` | string | `lang/topic`, e.g. `ar/belief`. |
| `metadata.modality` | string | `semantic` (spoken content) or `acoustic` (natural sound). |
| `metadata.task_type` | string | `completeness_fact` / `reasoning` / `time_localization` / `long_range`. |
| `metadata.judge_type` | string | Scoring route: `single_fact` / `reasoning` / `multi_slot` / `exhaustive_list` / `time_point_program` / `time_order`. |
| `metadata.difficulty` | string | `easy` / `hard`. |

### Loading

```python
import json
from collections import Counter
from pathlib import Path

root = Path("mula-bench")
with (root / "MuLA-Bench_v202609_public.jsonl").open(encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

# Expected inventory of the frozen evaluation set used in the paper.
assert len(rows) == len({r["id"] for r in rows}) == 5038
assert len({r["metadata"]["video_id"] for r in rows}) == 1732
assert Counter(r["metadata"]["modality"] for r in rows) == {
    "semantic": 3840,
    "acoustic": 1198,
}

row = rows[0]
qid = row["id"]
video_id = row["metadata"]["video_id"]
question = next(c["text"] for c in row["input"]["messages"][0]["content"]
                if c["type"] == "text")
gold = row["reference"]["answers"][0]
modality = row["metadata"]["modality"]   # "semantic" or "acoustic"
```

The model sees **audio + question only**. Do not pass `reference.evidence_text`,
`reference.answers`, or gold timestamps at inference time. The original question
may itself contain a time reference; preserve the question as released.

## Evaluation

The headline metric is **accuracy on the official 5,038 questions**, with a
**fixed denominator**. Unanswered items, refusals, and persistent input-limit
failures count as 0. Coverage (valid responses / 5,038) is diagnostic only.

### Fair input

1. **Canonical input** is the full source recording, except for the four
   predefined post-3 h questions from `Hgi8zNQnEa8`, which use the official fixed
   later-stage three-hour segment. Format / bitrate conversion is allowed if it
   does **not** drop content. Keep timestamps on the original source clock.
2. **Local models** receive the **longest prefix permitted by their native
   context limits, starting from the beginning of the canonical input**. The
   paper uses 3,276.8 s for MOSS-Audio-8B-Thinking, 2,400 s for
   Qwen3-Omni-30B-A3B-Instruct, and 1,800 s for Audio Flamingo Next 8B.
3. Outside the predefined canonical-input exception, evidence-aware or
   listen-point crops are not valid for the headline evaluation. Gold-aware
   crops are used only in separately reported diagnostics.

Slice scores use the official slice size as the denominator: **semantic 3,840**,
**acoustic 1,198**, **factual 1,891**, **reasoning 1,146**, **temporal 1,008**, and
**long-range 993**.

### Judge

The paper uses the frozen **`judge_v3`** end-to-end evaluator: **GPT-5.6 Sol**
checks semantic agreement under route-specific binary rubrics, and deterministic
guards enforce structural and temporal constraints. Only the **final committed
answer** is scored.

| Route (`judge_type`) | Questions | What is checked |
|---------------------|----------:|-----------------|
| `single_fact` | 2,454 | The correct core grounded fact. |
| `reasoning` | 1,145 | The target conclusion and the required relation or explanation. |
| `multi_slot` | 394 | Every mandatory slot and its correct entity–value assignment. |
| `exhaustive_list` | 81 | All required members; extra contradictory members fail. |
| `time_order` | 389 | A complete, correctly ordered, internally consistent event sequence under the stored item configuration. |
| `time_point_program` | 575 | Event match **plus** a programmatic clock check. |
| **Total** | **5,038** | |

Task families and judge routes are separate annotations, so their counts need not
match. The evaluator uses **low** reasoning effort for `single_fact` and
**medium** effort for the other five routes.

`time_point_program` uses **`TOL = 3.0 s`**:

- Gold point, predicted point: `abs(pred - gold) <= 3 s`.
- Gold interval, predicted point: the point falls inside `[g0 - 3, g1 + 3]`.
- Gold point, predicted interval: predicted width is **at most 10 s**, and the
  interval expanded by 3 s at each endpoint contains the gold point.
- Gold interval, predicted interval: predicted width is **at most 10 s**, and
  the predicted interval overlaps the gold interval expanded by 3 s at each endpoint.
- Predicted intervals **wider than 10 s** do not pass the temporal matcher.

Gold spans of at least **0.5 s**, or items explicitly typed as intervals, are
treated as reference intervals.

All **387 semantic temporal ordering questions** use `mention_order` with
`require_associated_times=false` and no per-item overrides: timestamp accuracy
is not required for credit on that slice. Follow the stored configuration for
other `time_order` items rather than applying this slice-specific setting to all
389 ordering-route questions.

The paper reports expert–evaluator agreement of **795 / 844 (94.2%)** across all
16 languages and all six judge routes. This is agreement on the stratified audit
sample, not a population-weighted estimate of judge accuracy.

Report the judge model, evaluator configuration, and temporal rules with any
published score. A different judge can change results on the LLM-judged routes.

### Scoring a prediction file

Save one JSONL row per official `id`:

```jsonl
{"id": "nonac|ar|belief|-aR600r70zQ|80b787bc2ad8", "response": "...model output..."}
```

(`response` may also be named `gen` or `prediction` in the supplied pipeline.)
Run the official `judge_v3` / `reeval_judge_v1` pipeline on the **5,038-id set**
using the frozen judge configuration. Items missing from the prediction file
are scored 0.

## License

**CC BY-NC 4.0** — released for **research, non-commercial use only**.
[License text](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en)

The **questions, answers, and audit labels** are released under CC BY-NC 4.0.
The **source recordings** remain social-media videos: `video_id` is provided so you
can fetch audio under the source platform’s terms. Redistributing raw social-media
audio may be restricted; do not treat the audio itself as CC BY-NC 4.0.

## Citation

If you use MuLA-Bench, please cite:

```bibtex
@misc{yang2026mulabench,
  title         = {{MuLA-Bench}: A Multilingual Long-Form Audio Understanding Benchmark via Multi-Tier Auditing},
  author        = {Zeyu Yang and Xinyu Zhang and Zibo Bi and Pei Zhang and Xize Cheng and Jin Xu and Baosong Yang and Satoshi Nakamura},
  year          = {2026},
  eprint        = {2609.23416},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SD},
  doi           = {10.48550/arXiv.2609.23416},
  url           = {https://arxiv.org/abs/2609.23416}
}
```
