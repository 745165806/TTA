"""H-UA1 source-anchored soft-mixture head math; source labels only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit


ALPHAS = (.25, .5, .75)


@dataclass(frozen=True)
class SourceGeometry:
    centroids: np.ndarray
    covariance: np.ndarray
    precision: np.ndarray
    ridge: float
    temperature: float
    w: np.ndarray
    b: float
    source_covariance_condition: float


def _distances(x: np.ndarray, centers: np.ndarray, precision: np.ndarray):
    diffs = x[:, None, :] - centers[None, :, :]
    return np.einsum("ncd,df,ncf->nc", diffs, precision, diffs, optimize=True)


def make_source_geometry(resources) -> SourceGeometry:
    anchors = resources.anchors_z.numpy().astype(np.float64)
    labels = resources.anchors_y.numpy().astype(np.int64)
    w = resources.w.numpy().astype(np.float64)
    if (anchors.ndim != 2 or anchors.shape[1] != 160 or w.shape != (160,) or
            set(labels.tolist()) != {0, 1} or not np.isfinite(anchors).all()):
        raise ValueError("invalid frozen source anchors/head")
    centroids = np.stack([anchors[labels == cls].mean(axis=0) for cls in (0, 1)])
    centered = anchors - centroids[labels]
    covariance = centered.T @ centered / (len(anchors) - 2)
    covariance = .5*(covariance + covariance.T)
    ridge = .1*float(np.trace(covariance))/160
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("nonpositive source covariance ridge")
    stable = covariance + ridge*np.eye(160)
    precision = np.linalg.inv(stable)
    anchor_distance = _distances(anchors, centroids, precision)
    temperature = float(np.median(np.abs(anchor_distance[:,0]-anchor_distance[:,1])))
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("source affinity temperature invalid")
    return SourceGeometry(centroids, covariance, precision, ridge, temperature,
                          w, float(resources.b), float(np.linalg.cond(stable)))


def adapt_buffer(views: np.ndarray, source: SourceGeometry):
    """Return label-free scores and auditable per-buffer diagnostic quantities."""
    values = np.asarray(views, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (3, 160) or not 1 <= len(values) <= 256:
        raise ValueError("H-UA1 requires a nonempty ≤256 buffer of production 3x160 views")
    if not np.isfinite(values).all():
        raise FloatingPointError("nonfinite target views")
    n = len(values)
    distances = _distances(values.reshape(-1, 160), source.centroids, source.precision)
    q_views = expit((distances[:,0]-distances[:,1])/source.temperature).reshape(n, 3)
    q1 = q_views.mean(axis=1)
    reliability = 1-(q_views.max(axis=1)-q_views.min(axis=1))
    if not (np.isfinite(q_views).all() and np.isfinite(reliability).all() and
            (reliability >= 0).all() and (reliability <= 1).all()):
        raise FloatingPointError("invalid continuous assignments or view reliability")
    q = np.stack((1-q1, q1), axis=1)
    weights = reliability[:, None]*q
    effective_counts = weights.sum(axis=0)
    if not np.isfinite(effective_counts).all() or np.any(effective_counts <= 1e-8):
        raise FloatingPointError("degenerate soft target class count")
    z = values[:,0,:]
    target_centroids = weights.T@z/effective_counts[:,None]
    prior = float(np.sum(reliability*q1)/np.sum(reliability))
    prior_clipped = float(np.clip(prior, 1e-4, 1-1e-4))
    centered = z[:,None,:]-target_centroids[None,:,:]
    target_covariance = np.einsum("nc,ncd,nce->de",weights,centered,centered,optimize=True)
    target_covariance /= np.sum(reliability)
    pooled = .5*source.covariance + .5*target_covariance + source.ridge*np.eye(160)
    pooled = .5*(pooled+pooled.T)
    chol = np.linalg.cholesky(pooled)
    delta = target_centroids[1]-target_centroids[0]
    w_raw = np.linalg.solve(chol.T,np.linalg.solve(chol,delta))
    raw_norm = float(np.linalg.norm(w_raw))
    if not np.isfinite(raw_norm) or raw_norm <= 1e-8:
        raise FloatingPointError("degenerate target LDA direction")
    b_raw = -.5*float(np.dot(target_centroids[0]+target_centroids[1],w_raw))
    b_raw += float(np.log(prior_clipped/(1-prior_clipped)))
    scale = float(np.linalg.norm(source.w))/raw_norm
    w_target, b_target = scale*w_raw, scale*b_raw
    frozen = z@source.w+source.b
    scores = {}
    for alpha in ALPHAS:
        w = (1-alpha)*source.w+alpha*w_target
        b = (1-alpha)*source.b+alpha*b_target
        scores[alpha] = z@w+b
    if not np.isfinite(frozen).all() or any(not np.isfinite(s).all() for s in scores.values()):
        raise FloatingPointError("nonfinite source-anchored head score")
    head_cos = float(np.dot(source.w,w_target)/(np.linalg.norm(source.w)*np.linalg.norm(w_target)))
    diagnostic = {"buffer_size":n, "source_temperature":source.temperature,
        "source_ridge":source.ridge, "source_covariance_condition":source.source_covariance_condition,
        "target_covariance_condition":float(np.linalg.cond(pooled)),
        "estimated_spoof_prior":prior,"prior_for_log":prior_clipped,
        "effective_class_counts":effective_counts.tolist(),
        "q_spoof_mean":float(q1.mean()),"q_spoof_min":float(q1.min()),
        "q_spoof_max":float(q1.max()),"reliability_mean":float(reliability.mean()),
        "reliability_min":float(reliability.min()),
        "target_head_cosine_to_source":head_cos,
        "target_head_angle_degrees":float(np.degrees(np.arccos(np.clip(head_cos,-1,1)))),
        "source_head_norm":float(np.linalg.norm(source.w)),
        "target_head_norm":float(np.linalg.norm(w_target)),
        "source_bias":source.b,"target_bias":float(b_target),
        "source_centroids":source.centroids.tolist(),
        "target_centroids":target_centroids.tolist(),
        "numeric_status":"ok"}
    return {"frozen":frozen,"scores":scores,"q_spoof":q1,
            "reliability":reliability,"diagnostic":diagnostic}
