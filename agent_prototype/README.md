# Minimal PIGuard Agent Prototype

This prototype places the reproduced PIGuard baseline detector before user input and
before tool output. It uses the existing checkpoint only:

`../logs/mof_baseline_20261023/checkpoints/best_model.pth`

The detector uses the original `PIGuard.classify()` interface, maximum sequence
length 256, and the existing argmax policy. No confidence threshold is applied.
Label `0` is benign and label `1` is injection. Detector failures fail closed.

## Run

From the project root:

```powershell
python agent_prototype\run_prototype.py
python agent_prototype\evaluate_prototype.py
python agent_prototype\evaluate_prototype.py --ollama
```

The default evaluation uses the deterministic mock LLM. `--ollama` uses the already
installed `qwen3:4b-nothink` model only when the local Ollama API reports it available.
If Ollama is unavailable, the run is recorded as `not_executed`; it never falls back
to MockLLM. No model download is performed.

Security events and result files are written under a new timestamped directory in
`agent_prototype\logs\`, separate from research logs. Existing evaluation outputs
are not overwritten. Tool responses are explicitly labeled untrusted, and blocked
tool output is never included in the LLM prompt.

## Metrics

For each stage independently:

- Detection rate = true positives / all injection cases.
- False-positive rate = false positives / all normal cases.
- False-negative rate = false negatives / all injection cases.
- Attack Success Rate (ASR) = independently verified objective successes /
  evaluated injection cases. Tool-error cases are excluded because no tool output
  was produced, and are reported separately.
- Normal Task Completion Rate = fixture-defined expected outcomes verified /
  all normal cases.

An allowed attack is not automatically successful. The fixture uses exact protected
secret disclosure evidence, not keyword presence. If that objective cannot be
independently verified, it is reported as `unverified` and remains in the ASR
denominator. The MockLLM and Ollama runs are stored in separate timestamped
directories and identified in `summary.json`.

Tool and LLM failures are reported explicitly and are not counted as task completions
or attack successes.

## Autonomous Ollama agent

The separate `interactive_agent.py` uses Ollama to select tools autonomously. It
requires strict JSON decisions using only `calculator` and `mock_search`, validates
all arguments, and permits at most three tool calls per request. User input and
every tool output are checked by PIGuard before entering LLM context. Ollama is
never silently replaced with MockLLM.

```powershell
python agent_prototype\interactive_agent.py
```

The existing `run_prototype.py` and `evaluate_prototype.py` retain their original
fixture-driven interfaces and are not replaced by the interactive runner.

## Limitations

This is an academic prototype, not a production security boundary. It uses a small
deterministic tool fixture, a simple mock LLM, and a single baseline detector. The
agent does not implement a full tool schema, authorization system, sandbox, prompt
isolation protocol, or calibrated confidence policy.
