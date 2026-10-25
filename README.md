# tool-loop

[![tests](https://github.com/tiantianw/tool-loop/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/tiantianw/tool-loop/actions/workflows/test.yml)

A tool-calling agent in one readable loop. **English** · [中文](README.zh-CN.md)

Most agent frameworks hide the part that matters. You install one and get a
loop, but the failure handling is spread across a dozen modules and the
reasoning behind it is invisible. This repo is the opposite: one loop, three
read-only tools, two dependencies, and a README that says exactly what happens
when a tool fails.

The loop is `agent.py`, about 280 lines, a third of which is comment explaining
*why* rather than *what*.

```
python demo.py        # no API key, no network
pytest                # 36 tests
```

---

## Try it without an API key

```bash
pip install httpx
python demo.py
```

The demo swaps the model for a script, so it runs offline and prints the same
output every time. It walks into four failures on purpose and recovers from
each one:

```
step 1: read_file  src/agent.py   -> not_found: no such file
step 2: list_dir   src            -> not_found: no such directory
step 3: list_dir   .              -> agent.py, llm.py, tools.py, ...
step 4: grep       "class ToolError["  -> invalid_argument: unterminated character set
step 5: grep       "class ToolError"   -> agent.py:35: class ToolError(Exception):
step 6: read_file  agent.py       -> lines 1-200 of 277 (truncated)
step 7: answer
```

Steps 1–2 are the interesting pair: the hint from step 1 points at `src`, which
does not exist either, so the model has to walk back one more level. A single
hint is not a fix, and the loop does not pretend otherwise.

The full trace is in [examples/transcript.md](examples/transcript.md).

## The one decision that matters

> **A tool failure is data, not an exception.**

The model can fix a wrong path, a malformed regex, an argument of the wrong
type. It cannot fix a `429` or an expired key. Those are different problems and
they belong to different layers:

- **Transport failures** (429, 5xx, timeouts) are retried with exponential
  backoff and jitter in `llm.py`. The loop never sees them. If retries run out,
  the exception is fatal and the run stops.
- **Tool failures** (missing file, bad regex, bad arguments) are serialised into
  the tool result and handed back to the model. The run continues, and the model
  decides whether to retry, change approach, or give up.

If tool errors were raised instead of returned, none of the recovery tests in
`tests/test_recovery.py` could finish. That is the whole argument for the design.

## How the loop is built

```
                 ┌──────────────────┐
                 │  build messages  │◄──────────────┐
                 └────────┬─────────┘               │
                          ▼                         │
                 ┌──────────────────┐               │
                 │    call model    │  retry 429/5xx│
                 └────────┬─────────┘  in llm.py    │
                          ▼                         │
                 ┌──────────────────┐               │
                 │  parse response  │               │
                 └────┬────────┬────┘               │
        no tool_calls │        │ tool_calls         │
                      ▼        ▼                    │
              ┌──────────┐  ┌──────────────────┐    │
              │  answer  │  │ execute tools    │────┘
              └──────────┘  │ errors → results │
                            └──────────────────┘
```

| file | lines | what lives there |
|---|---|---|
| `agent.py` | ~280 | the loop, `ToolRegistry`, argument validation, loop-level guards |
| `llm.py` | ~190 | HTTP transport, retry policy, `ScriptedLLM` |
| `tools.py` | ~220 | `list_dir`, `read_file`, `grep`, and their failure modes |
| `main.py` | ~50 | CLI |
| `demo.py` | ~110 | scripted run and transcript renderer |

## Failure handling

| layer | failure | handled by | what happens |
|---|---|---|---|
| transport | 429, 5xx, socket reset, timeout | `llm.py` | retry with backoff + jitter, up to 3 attempts |
| transport | 401, 400, unknown model | `llm.py` | no retry, `FatalError` immediately |
| transport | response body is not valid JSON | `llm.py` | treated as transient, retried |
| protocol | `tool_calls` entry with no name | `llm.py` | treated as transient, retried |
| tool | arguments are not valid JSON | `ToolRegistry` | `bad_json` result with a hint about code fences |
| tool | missing or mistyped field | `ToolRegistry` | `invalid_arguments`, naming the offending field |
| tool | unknown tool name | `ToolRegistry` | `unknown_tool`, listing the real ones |
| tool | file or directory missing | `tools.py` | `not_found`, hinting which directory to list |
| tool | path escapes the workspace | `tools.py` | `invalid_argument`, refused before touching disk |
| tool | file is binary | `tools.py` | `invalid_argument`, suggests `grep` instead |
| tool | file too large | `tools.py` | truncated, header reports the real line count |
| tool | regex does not compile | `tools.py` | `invalid_argument`, carrying the `re` message |
| tool | no matches | `tools.py` | a normal result, not an error |
| tool | the tool itself crashes | `ToolRegistry` | `tool_crashed`, "repeating this will not help" |
| loop | identical call repeated | `agent.py` | 3rd repeat stops the run and asks for a summary |
| loop | step limit | `agent.py` | forced final answer, tools withheld |
| loop | token budget | `agent.py` | forced final answer, tools withheld |

Two details are easy to miss and worth calling out:

**Validation names the field.** `"'start_line' must be integer, got str"` is
recoverable in one turn. `"invalid arguments"` is not — the model has to guess.

**Stopping withholds tools.** `_finalize()` passes `tools=None` rather than
asking the model to please stop. A tool call becomes impossible, so an answer is
the only remaining move.

## Four decisions, and what each one costs

**No framework.** No LangChain, no LlamaIndex. *Cost:* no streaming, no memory,
no multi-agent, no tracing UI. *Reason:* those are the parts you can add once the
loop is correct, and the loop is the part that is usually wrong.

**Hand-rolled argument validation.** About 30 lines covering required keys and
primitive types, instead of a JSON Schema library. *Cost:* no `oneOf`, no
`format`, no nested object validation. *Reason:* models make two mistakes —
omitting a field and passing the wrong type. Both are covered, and the error
message can be written for the model to read rather than for a log.

**The scripted model is a first-class citizen.** `ScriptedLLM` is not a test
fixture bolted on at the end; it is how the demo and all 36 tests run. *Cost:*
the scripts need maintenance when behaviour changes. *Reason:* if you cannot
reproduce a failure without spending money, you will not test it.

**Read-only tools.** `list_dir`, `read_file`, `grep`. *Cost:* the agent cannot
change anything, so this is not a coding agent. *Reason:* a public repo that can
write files is a public repo someone runs against their home directory. Read-only
still produces every interesting failure — missing paths, permissions, oversized
files, encoding, bad regex — without the blast radius.

## Using a real model

Any OpenAI-compatible endpoint works:

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini          # optional
export OPENAI_BASE_URL=https://api.openai.com/v1   # optional

python main.py "which module handles retries, and why there?" --root .
```

The code calls `/chat/completions` with `httpx` directly. No SDK, so nothing
hides the request shape and no provider is baked in.

## Tests

```
pytest -q
36 passed
```

| file | covers |
|---|---|
| `tests/test_loop.py` | termination, step limit, repeat detection, token budget, history shape |
| `tests/test_failures.py` | every error payload, plus retry, backoff growth, 401 vs 429 |
| `tests/test_recovery.py` | the model being wrong, reading the error, and correcting itself |

`tests/test_failures.py` patches `httpx.Client.post`, so retry behaviour is
tested without a network or a sleep.

## Limits

Stated plainly, because a sample that claims to do everything is a sample nobody
believes:

- Tools run **serially**. A model asking for three reads pays for three round
  trips.
- The message history **grows without bound**. A long task will eventually
  exceed the context window, and there is no compaction.
- No streaming, so nothing appears until the turn completes.
- Retry count is fixed at 3. It does not adapt to the remaining token budget.
- One workspace, one agent, no persistence. If the process dies, the run is lost.
- `grep` walks the tree with `rglob` and caps at 2,000 files. It is a demo tool,
  not ripgrep.

## What production would need

- **Structured traces**, not `print`. One span per model call and per tool call,
  with token counts and latency, exported to OTel.
- **Idempotency keys** on tool calls, so a retried step does not repeat a side
  effect.
- **Approval gates** for anything that mutates state, and a real sandbox for
  anything that executes.
- **Context compaction** — summarise old turns before the window fills, rather
  than discovering the limit at runtime.
- **Concurrent tool execution** for independent calls.
- **An eval set.** The recovery tests here prove the mechanism works; they do
  not prove a given model will use it.

## License

MIT
