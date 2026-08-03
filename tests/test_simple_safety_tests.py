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
