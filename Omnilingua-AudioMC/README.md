# Omnilingua-AudioMC

Multilingual multi-turn **spoken-dialogue evaluation** with rubric-based LLM
grading, in 10 languages. This is the benchmark introduced in our paper *"Do
Multi-Turn Conversational Capabilities Carry Over from Text to Speech? A
Parallel Multilingual Benchmark"* (coming soon on arXiv): a localized
derivative of **Audio MultiChallenge (AudioMC)**
([arXiv:2512.14865](https://arxiv.org/abs/2512.14865),
[ScaleAI/audiomc](https://huggingface.co/datasets/ScaleAI/audiomc)), released
as a subset of [Omnilingua-Bench](https://github.com/QwenLM/Omnilingua-Bench).

A model is dropped into a multi-turn voice conversation and scored on its
**final** assistant reply of each conversation, across 4 axes:

| axis | what it measures |
|---|---|
| `INFERENCE_MEMORY` | recalling facts stated earlier in the conversation |
| `INSTRUCTION_RETENTION` | honoring constraints given in earlier turns |
| `SELF_COHERENCE` | staying consistent with the model's own previous replies |
| `VOICE_EDITING` | rewriting a previous reply under a new instruction |

This directory ships the **benchmark data** (conversation indexes + original
audio), the **judge prompt**, and the full **evaluation protocol** needed to
score a model and get comparable numbers. The user-turn audio (~27 GB of WAV)
is distributed as a separate dataset archive; drop it at `data/<lang>/audio/`.

## Dataset

```
data/
└── <lang>/
    ├── index.jsonl        # one conversation per line
    └── audio/             # user-turn WAV clips referenced by the index
prompts/
└── judge.md               # judge prompt (verbatim from the AudioMC paper)
```

| axis | conversations |
|---|---|
| `INFERENCE_MEMORY` | 82 |
| `INSTRUCTION_RETENTION` | 95 |
| `SELF_COHERENCE` | 77 |
| `VOICE_EDITING` | 100 |

Released languages — identical conversation structure across languages
(354 conversations / 1,404 rubric items / 1,827 audio clips each):


| language | audio format | notes |
|---|---|---|
| `en` | WAV, 48 kHz mono | original English recordings (upstream) |
| `en_tts` | WAV, 24 kHz mono | English answered on synthesized audio — the TTS-vs-original control arm |
| `zh` `ja` `es` `fr` `ru` `de` `it` `th` `vi` | WAV, 24 kHz mono | localized conversations |

Conversations with the same `source_conv_id` across languages are parallel
instances of one source conversation, so scores are directly comparable across
languages.

Index schema (one record per line):

```json
{"conv_id": "xx_ID123", "source_conv_id": "ID123", "lang": "xx",
 "axis": "VOICE_EDITING", "is_audio_cue": false,
 "turns": [{"turn": 1, "wav": "ID123_turn1.wav",
            "user_transcript": "...", "assistant_transcript": "..."}],
 "rubric": ["..."], "meta": {"register": "casual"}}
```

- `wav` is a bare filename resolved against the language's `audio/` directory.
- The last turn's `assistant_transcript` is empty — that slot is the model's
  to fill.
- `register` is the conversation's politeness register (`casual` / `polite`).

**Differences from the upstream English release.** This is a derivative of the
upstream 452-conversation English benchmark (MIT), with: (1) localization into
9 additional language arms; (2) quality control — mistranslated or broken
items removed and the corresponding audio re-synthesized; (3) Audio-Cue conversations excluded entirely, since they
hinge on background sounds that localized audio does not carry.

## Evaluation protocol

### 1. Serving the model

Each conversation is sent to the model under test as one multi-turn chat
request — the audio setting from the paper:

```
system    : "You are a helpful voice assistant in a spoken conversation.
             Respond naturally and concisely to the user's latest turn."
user      : [audio of user turn 1]        (input_audio content part, base64)
assistant : dataset reference transcript for turn 1
...
user      : [audio of the last user turn]  <- the model answers HERE
```

- Every user turn is sent as audio (`{"type": "input_audio", "input_audio":
  {"data": <base64>, "format": "wav"}}` in an OpenAI-compatible chat API);
  previous assistant turns are the dataset's reference transcripts. The model
  produces only the final reply, which is what the rubrics grade.
- The reply is graded as text: speech-to-speech systems are evaluated on their
  text response.




### 2. Grading (LLM-as-a-Judge)

The judge is a strong **text-only** model (we use `openai/o4-mini`; do not use
a model that is itself under test, or from the same family as one under test).
One call per rubric item: the judge sees the transcript history with the
model's reply appended, plus the rubric item, and returns
`{"explanation", "criteria_met"}`. The prompt is
[`prompts/judge.md`](prompts/judge.md), verbatim from the paper, kept in
English for every language so the grading standard is comparable.

Build the transcript history with the model's reply as the final turn:

```python
def build_grading_conversation_history(record: dict, model_response: str) -> str:
    """Format the transcript history for the judge, with the model's reply last."""
    parts = []
    for i, turn in enumerate(record["turns"]):
        if turn["user_transcript"].strip():
            parts.append(f"User: {turn['user_transcript'].strip()}")
        if i < len(record["turns"]) - 1:
            if turn["assistant_transcript"].strip():
                parts.append(f"Assistant: {turn['assistant_transcript'].strip()}")
        else:
            parts.append(f"Assistant: {model_response.strip()}")
    return "\n\n".join(parts)
```

### 3. Aggregation

Two metrics per (model, language), both from the per-rubric `criteria_met`
verdicts:

- **Instance pass rate** (the headline number): a conversation passes only
  when *every* one of its rubric items is met.
- **Rubric pass rate** (diagnostic): fraction of individual rubric items met.

A conversation whose inference failed, or whose rubrics were not all graded,
is **incomplete** — excluded from the rates, not scored as a failure. Report
the incomplete count alongside; a healthy run scores 354 / 354 conversations
per language.

## License

This dataset is intended solely for academic research and model evaluation purposes and must not be used for any commercial products or services. The third-party model API outputs included in the dataset were generated based on the publicly available service agreements of the respective model or service providers; intellectual property rights and usage permissions regarding the content of these outputs are governed by those agreements. Users are prohibited from utilizing this dataset to train, develop, or improve any artificial intelligence models.

The index and text annotations in this directory are released under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en).
The task design, judge prompt, and the English source conversations derive
from [ScaleAI/audiomc](https://huggingface.co/datasets/ScaleAI/audiomc)
(MIT, © the Audio MultiChallenge authors). The localized conversations and
synthesized audio (`en_tts` and the non-English arms) were produced for this
release.

## Citation

If you use this benchmark, please cite our paper that introduces it
(coming soon on arXiv):

```bibtex
@misc{omnilingua_audiomc_2026,
  title        = {Do Multi-Turn Conversational Capabilities Carry Over from Text to Speech? A Parallel Multilingual Benchmark},
  year         = {2026},
  note         = {Coming soon on arXiv}
}
```

This subset is part of Omnilingua-Bench:

```bibtex
@misc{omnilingua_bench_2026,
  title        = {Omnilingua-Bench: Multilingual Omni-modal Evaluation of Long Audio Recognition, Multi-speaker Understanding, and Instruction Following},
  author       = {{Alibaba Token Hub, Alibaba Group}},
  year         = {2026},
  howpublished = {\url{https://github.com/QwenLM/Omnilingua-Bench}}
}
```

The task design and judge protocol follow Audio MultiChallenge:

```bibtex
@misc{gosai2025audiomultichallenge,
  title        = {Audio MultiChallenge: A Multi-Turn Evaluation of Spoken Dialogue Systems on Natural Human Interaction},
  author       = {Advait Gosai and Tyler Vuong and Utkarsh Tyagi and Steven Li and Wenjia You and Miheer Bavare and Arda U{\c{c}}ar and Zhongwang Fang and Brian Jang and Bing Liu and Yunzhong He},
  year         = {2025},
  eprint       = {2512.14865},
  archivePrefix= {arXiv},
  primaryClass = {cs.SD}
}
```
