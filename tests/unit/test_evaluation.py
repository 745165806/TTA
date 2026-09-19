import pytest

from eptta.evaluation.metrics import binary_metrics
from eptta.errors import DataError


def test_metrics_preserve_label_and_score_direction():
    result = binary_metrics([-2.0, -1.0, 1.0, 2.0], [0, 0, 1, 1], 0.0,
                            frozen_scores=[2.0, -1.0, -1.0, 2.0])
    assert result["auroc"] == result["balanced_accuracy"] == 1.0
    assert result["eer"] == 0.0
    assert result["helpful_flips"] == 2
    assert result["helpful_flips_by_class"] == {"0": 1, "1": 1}
    assert result["harmful_flips_by_class"] == {"0": 0, "1": 0}


@pytest.mark.parametrize("scores,labels", [([0.0], [0]), ([float("nan"), 1.0], [0, 1]),
                                             ([0.0, 1.0], [0, 2])])
def test_metrics_reject_invalid_boundaries(scores, labels):
    with pytest.raises((DataError, ValueError)):
        binary_metrics(scores, labels, 0.0)
