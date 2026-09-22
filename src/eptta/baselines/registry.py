"""Complete method contracts layered over the user-facing status catalog."""
from eptta.errors import ContractError
from eptta.registry import get_spec


_DETAILS = {
    "frozen": ("none", "none", "none", "none", "closed_form", (), "original"),
    "multiview_mean": ("none", "none", "none", "none", "closed_form", (), "mean_views"),
    "static_subspace": ("shared_matrix", "none", "none", "response", "source_selected_grid", ("U",), "original"),
    "fixed_source_adapter": ("shared_matrix", "source_view_variance", "margin", "response", "offline_sgd", ("U", "M"), "original"),
    "frozen_source_shift": ("scalar_bias", "none", "none", "none", "source_selected_grid", (), "original"),
    "ep_tta": ("matrix", "view_variance", "margin", "response", "sgd", ("U", "M"), "original"),
    "ep_tta_guarded": ("matrix", "view_variance", "margin", "response", "sgd_margin_guard", ("U", "M"), "original"),
    "ep_no_keep": ("matrix", "view_variance", "none", "response", "sgd", ("U",), "original"),
    "ep_random_U": ("matrix", "view_variance", "margin", "random", "sgd", ("U_random", "M"), "original"),
    "ep_feature_pca_U": ("matrix", "view_variance", "margin", "feature_pca", "sgd", ("U_feature_pca", "M"), "original"),
    "ep_no_projection": ("matrix", "view_variance", "margin", "response", "sgd_unprojected", ("U", "M"), "original"),
    "entropy_same_adapter_no_keep": ("matrix", "mean_view_entropy", "none", "response", "sgd", ("U",), "original"),
    "entropy_same_adapter": ("matrix", "mean_view_entropy", "margin", "response", "sgd", ("U", "M"), "original"),
    "memo_same_adapter_no_keep": ("matrix", "marginal_entropy", "none", "response", "sgd", ("U",), "original"),
    "memo_same_adapter_keep": ("matrix", "marginal_entropy", "margin", "response", "sgd", ("U", "M"), "original"),
    "ep_keep_l2": ("matrix", "view_variance", "parameter_l2", "response", "sgd", ("U",), "original"),
    "ep_keep_logit": ("matrix", "view_variance", "source_logit", "response", "sgd", ("U", "M"), "original"),
    "ep_keep_fisher": ("matrix", "view_variance", "fisher", "response", "sgd", ("U", "F"), "original"),
    "ep_scalar_adaptive": ("negative_scalar_identity", "view_variance", "margin", "response", "fixed_17_grid", ("U", "M"), "original"),
    "ep_diagonal_R": ("diagonal", "view_variance", "margin", "response", "sgd", ("U", "M"), "original"),
    "source_ce_only": ("matrix", "none", "source_ce", "response", "sgd", ("U", "M"), "original"),
    "tent_audio_ep": ("author_model_parameters", "author_contract_pending", "author_contract_pending", "n/a", "author_contract_pending", (), "original"),
    "sar_audio_ep": ("author_model_parameters", "author_contract_pending", "author_contract_pending", "n/a", "author_contract_pending", (), "original"),
    "memo_audio_ep_full": ("author_model_parameters", "author_contract_pending", "author_contract_pending", "n/a", "author_contract_pending", (), "original"),
    "eata_audio_ep": ("author_model_parameters", "author_contract_pending", "author_contract_pending", "n/a", "author_contract_pending", (), "original"),
    "t2a_audio_ep": ("author_model_parameters", "author_contract_pending", "author_contract_pending", "n/a", "author_contract_pending", (), "original"),
}


def get_method_contract(method_id):
    status = get_spec("methods", method_id)
    try:
        parameterization, objective, regularizer, subspace, solver, resources, final_output = _DETAILS[method_id]
    except KeyError as exc:
        raise ContractError("method has status registration but no complete contract: %s" % method_id) from exc
    return {"method_id": method_id, "comparison_track": status["comparison_track"], "route": status["route"],
            "parameterization": parameterization, "objective": objective, "regularizer": regularizer,
            "subspace": subspace, "solver": solver, "source_resources": resources,
            "reset_policy": "per_sample", "final_output": final_output,
            "implementation_status": status["implementation_status"]}
