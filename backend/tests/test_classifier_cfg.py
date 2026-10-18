import pytest
from app.schemas.schemas import Verdict
from app.services.classifier import resolve_fake_index, verdict_from_prob


def test_pretrained_label_0_is_fake_by_override():
    # jy46604790/Fake-News-Bert-Detect: LABEL_0 = fake, LABEL_1 = real
    assert resolve_fake_index({0: "LABEL_0", 1: "LABEL_1"}, "LABEL_0") == 0
    assert resolve_fake_index({"0": "LABEL_0", "1": "LABEL_1"}, "0") == 0


def test_named_fake_label_autodetected():
    assert resolve_fake_index({0: "REAL", 1: "FAKE"}) == 1
    assert resolve_fake_index({0: "fake", 1: "real"}) == 0


def test_default_is_index_one_and_bad_override_errors():
    assert resolve_fake_index({0: "LABEL_0", 1: "LABEL_1"}) == 1
    with pytest.raises(ValueError):
        resolve_fake_index({0: "A", 1: "B"}, "FAKE")


def test_classifier_commits_only_when_confident():
    assert verdict_from_prob(0.97, 0.85) == Verdict.likely_fake
    assert verdict_from_prob(0.03, 0.85) == Verdict.likely_real
    assert verdict_from_prob(0.70, 0.85) == Verdict.uncertain     # would have been "fake" under the old 0.65 rule
    assert verdict_from_prob(0.30, 0.85) == Verdict.uncertain
