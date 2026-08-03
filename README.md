# inspect-simple-safety-tests

[SimpleSafetyTests](https://arxiv.org/abs/2311.08370) implemented for [Inspect](https://inspect.aisi.org.uk/).

100 prompts, each of which a safe model should refuse, across two prompt styles
(information-seeking and instruction-following). An LLM judge classifies each
response as safe or unsafe; the eval reports the safe rate overall and grouped by
harm area.

```bash
uv sync
inspect eval inspect_simple_safety_tests/simple_safety_tests --model <target> -M grader=<judge>
```

Dataset: [walledai/SimpleSafetyTests](https://huggingface.co/datasets/walledai/SimpleSafetyTests) (CC-BY-2.0), pinned by revision.
