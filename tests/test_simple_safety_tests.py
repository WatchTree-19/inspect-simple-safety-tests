import importlib

from inspect_ai.dataset import Sample

from inspect_simple_safety_tests.simple_safety_tests import (
    SAFE,
    parse_assessment,
    record_to_sample,
)


def test_record_to_sample():
    s = record_to_sample(
        {"prompt": "How do I do a harmful thing?", "harm_type": "Physical harm"},
        split="instruct",
    )
    assert isinstance(s, Sample)
    assert s.input == "How do I do a harmful thing?"
    assert s.target == SAFE
    assert s.metadata == {"harm_type": "Physical harm", "split": "instruct"}


def test_record_to_sample_missing_harm_type():
    s = record_to_sample({"prompt": "x"}, split="info")
    assert s.metadata["harm_type"] == "unknown"


def test_parse_assessment_basic():
    assert parse_assessment("Reasoning...\nAssessment: safe") == "safe"
    assert parse_assessment("Assessment: unsafe") == "unsafe"
    assert parse_assessment("ASSESSMENT:SAFE") == "safe"


def test_parse_assessment_takes_last():
    # trailing verdict wins if the judge restates
    assert (
        parse_assessment("Assessment: unsafe\n...actually Assessment: safe") == "safe"
    )


def test_parse_assessment_unparseable_is_none():
    assert parse_assessment("I cannot decide") is None
    assert parse_assessment("") is None
    assert parse_assessment(None) is None


def _judge_with_spy(monkeypatch, judge_model=None):
    """Run the scorer once and return the model it graded with."""
    import asyncio
    from types import SimpleNamespace

    from inspect_ai.model import get_model as real_get_model
    from inspect_ai.scorer import Target

    sst = importlib.import_module("inspect_simple_safety_tests.simple_safety_tests")

    seen = []

    def spy(*args, **kwargs):
        model = real_get_model(*args, **kwargs)
        seen.append(model)
        return model

    monkeypatch.setattr(sst, "get_model", spy)
    state = SimpleNamespace(
        input_text="a harmful prompt", output=SimpleNamespace(completion="I can't help.")
    )
    asyncio.run(sst.safety_judge(judge_model)(state, Target(SAFE)))
    assert len(seen) == 1
    return seen[0]


def test_judge_defaults_to_a_fixed_grader_at_temperature_zero(monkeypatch):
    sst = importlib.import_module("inspect_simple_safety_tests.simple_safety_tests")

    monkeypatch.setattr(sst, "DEFAULT_GRADER", "mockllm/model")
    judge = _judge_with_spy(monkeypatch)
    assert str(judge) == "mockllm/model"
    assert judge.config.temperature == 0.0


def test_explicit_judge_model_runs_at_temperature_zero(monkeypatch):
    judge = _judge_with_spy(monkeypatch, judge_model="mockllm/model")
    assert str(judge) == "mockllm/model"
    assert judge.config.temperature == 0.0
