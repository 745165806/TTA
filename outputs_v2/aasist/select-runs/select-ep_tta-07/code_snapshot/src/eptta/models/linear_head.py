"""Semantic conversion of an audited two-logit author head."""
from eptta.errors import ContractError


def export_spoof_score_head(weight, bias, class_index_map):
    if set(class_index_map) != {"bonafide", "spoof"} or set(class_index_map.values()) != {0, 1}:
        raise ContractError("class index map must be a bijection")
    if getattr(weight, "ndim", None) != 2 or weight.shape[0] != 2:
        raise ContractError("author head weight must have shape [2,d]")
    if getattr(bias, "ndim", None) != 1 or bias.shape[0] != 2:
        raise ContractError("author head bias must have shape [2]")
    fake = class_index_map["spoof"]
    real = class_index_map["bonafide"]
    return weight[fake] - weight[real], bias[fake] - bias[real]


def score_embedding(embedding, weight, bias):
    return embedding.matmul(weight) + bias
