"""Deterministic label-free two-component one-dimensional Gaussian mixture."""
from dataclasses import asdict, dataclass
import math

import numpy as np


class CalibrationFitError(RuntimeError):
    code = "CALIBRATION_FIT_FAIL"


@dataclass(frozen=True)
class GaussianMixture1D:
    mu_bona: float
    mu_spoof: float
    var_bona: float
    var_spoof: float
    pi_bona: float
    pi_spoof: float
    tau_hat: float
    converged: bool
    iterations: int

    def posterior(self, score):
        values = np.asarray(score, dtype=np.float64)
        if not np.isfinite(values).all():
            raise CalibrationFitError("nonfinite score passed to posterior")
        log_bona = (math.log(self.pi_bona) - 0.5 * math.log(self.var_bona) -
                    0.5 * (values - self.mu_bona) ** 2 / self.var_bona)
        log_spoof = (math.log(self.pi_spoof) - 0.5 * math.log(self.var_spoof) -
                     0.5 * (values - self.mu_spoof) ** 2 / self.var_spoof)
        delta = np.clip(log_bona - log_spoof, -700.0, 700.0)
        result = 1.0 / (1.0 + np.exp(delta))
        return float(result) if result.ndim == 0 else result

    def as_dict(self):
        return asdict(self)


def _crossing(mu_bona, mu_spoof, var_bona, var_spoof, pi_bona, pi_spoof):
    def log_odds_spoof(x):
        return (math.log(pi_spoof) - 0.5 * math.log(var_spoof) -
                0.5 * (x - mu_spoof) ** 2 / var_spoof -
                math.log(pi_bona) + 0.5 * math.log(var_bona) +
                0.5 * (x - mu_bona) ** 2 / var_bona)

    low, high = mu_bona, mu_spoof
    f_low, f_high = log_odds_spoof(low), log_odds_spoof(high)
    if not all(math.isfinite(v) for v in (f_low, f_high)) or f_low > 0 or f_high < 0:
        raise CalibrationFitError("component posterior has no bona-to-spoof crossing between means")
    for _ in range(100):
        mid = 0.5 * (low + high)
        value = log_odds_spoof(mid)
        if value >= 0:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


def fit_gmm(scores, *, max_iter=100, tol=1e-6, variance_floor=1e-6):
    """Fit a GMM from scores only; this API intentionally has no labels argument."""
    x = np.asarray(scores, dtype=np.float64)
    if x.ndim != 1 or x.size < 10 or not np.isfinite(x).all():
        raise CalibrationFitError("scores must be a finite 1-D array with at least 10 values")
    if type(max_iter) is not int or max_iter <= 0 or tol <= 0 or variance_floor <= 0:
        raise ValueError("invalid deterministic GMM configuration")
    global_var = float(np.var(x))
    if not math.isfinite(global_var) or global_var <= variance_floor:
        raise CalibrationFitError("variance collapse")
    means = np.asarray(np.quantile(x, [0.25, 0.75]), dtype=np.float64)
    if means[1] - means[0] <= max(1e-6, 1e-3 * math.sqrt(global_var)):
        raise CalibrationFitError("means indistinguishable")
    variances = np.asarray([global_var, global_var], dtype=np.float64)
    weights = np.asarray([0.5, 0.5], dtype=np.float64)
    previous_ll = None
    converged = False
    iterations = 0
    for iteration in range(1, max_iter + 1):
        log_prob = np.stack([
            np.log(weights[k]) - 0.5 * np.log(2.0 * math.pi * variances[k]) -
            0.5 * (x - means[k]) ** 2 / variances[k]
            for k in range(2)
        ], axis=1)
        row_max = np.max(log_prob, axis=1, keepdims=True)
        normalizer = row_max + np.log(np.exp(log_prob - row_max).sum(axis=1, keepdims=True))
        responsibilities = np.exp(log_prob - normalizer)
        ll = float(normalizer.sum())
        effective = responsibilities.sum(axis=0)
        if not np.isfinite(responsibilities).all() or np.any(effective < 2.0):
            raise CalibrationFitError("component collapse")
        new_weights = effective / x.size
        new_means = (responsibilities * x[:, None]).sum(axis=0) / effective
        raw_variances = (responsibilities * (x[:, None] - new_means) ** 2).sum(axis=0) / effective
        if (not np.isfinite(raw_variances).all() or np.any(raw_variances <= variance_floor) or
                np.any(new_weights <= 1e-3)):
            raise CalibrationFitError("variance or component collapse")
        variances = np.maximum(raw_variances, variance_floor)
        means, weights = new_means, new_weights
        iterations = iteration
        if previous_ll is not None and abs(ll - previous_ll) <= tol:
            converged = True
            break
        previous_ll = ll
    if not converged:
        raise CalibrationFitError("EM non-convergence")
    order = np.argsort(means)
    means, variances, weights = means[order], variances[order], weights[order]
    if means[1] - means[0] <= max(1e-6, 1e-3 * math.sqrt(global_var)):
        raise CalibrationFitError("means indistinguishable")
    tau_hat = _crossing(means[0], means[1], variances[0], variances[1],
                        weights[0], weights[1])
    return GaussianMixture1D(
        mu_bona=float(means[0]), mu_spoof=float(means[1]),
        var_bona=float(variances[0]), var_spoof=float(variances[1]),
        pi_bona=float(weights[0]), pi_spoof=float(weights[1]),
        tau_hat=float(tau_hat), converged=True, iterations=iterations,
    )
