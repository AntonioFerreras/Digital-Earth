"""Error metrics of a predictor against the clean reference points, overall and by region.

Predictions are clamped at zero first: the shipped head ends in ReLU
(shaders/neural/mlp.slang), so that is what the renderer shows.
"""

import numpy as np

from study import view_uvwz

# Relative errors treat targets below about sqrt(EPS) = 0.01 (1% of the median
# radiance, far below a lit sky) as absolute: the night side is ~1e-5 and its
# relative error is invisible at any exposure the renderer uses.
EPS = 1e-4
# Points whose mean reference RGB exceeds this are "lit" for the relative-error statistics.
LIT_MIN = 0.01

def regions(uvwz: np.ndarray) -> dict[str, np.ndarray]:
    """Masks of the hard parts of the domain."""
    _, _, mu_s, _ = view_uvwz.scattering_params_from_uvwz(uvwz)
    return {
        # u_mu = 0.5 splits rays that hit the ground (straight down) from those that
        # do not (straight up), a discontinuity; grazing rays sit at both ends.
        "split": np.abs(uvwz[:, 2] - 0.5) < 0.02,
        "horizon": np.abs(uvwz[:, 2] - 0.5) > 0.48,
        "terminator": np.abs(mu_s) < 0.05,
        "forward": uvwz[:, 0] > 0.995,  # nu > 0.99
        "sunset": (mu_s > -0.05) & (mu_s < 0.15),
    }


def evaluate(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Metrics of `prediction` [N, 3] against reference records [N, 7] (uvwz, RGB).

    Points no camera ray reaches (view_uvwz.reachable) are left out. Relative errors are per channel value over the lit values (reference above
    LIT_MIN): a channel near zero beside two bright ones has no meaningful
    relative error, and the absolute terms (rel_l2, RMSE) cover it."""
    assert prediction.shape == (reference.shape[0], 3)
    assert reference.shape[1] == 7
    inside = view_uvwz.reachable(reference[:, :4])
    prediction, reference = prediction[inside], reference[inside]
    p = np.maximum(prediction.astype(np.float64), 0.0)
    y = reference[:, 4:].astype(np.float64)
    lit = y > LIT_MIN
    relative = np.abs(p - y) / np.maximum(np.abs(y), LIT_MIN)
    metrics = {
        "rel_l2": float(np.mean((p - y) ** 2 / (y * y + EPS))),
        "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
        "lit_fraction": float(lit.mean()),
        "rel_mean": float(relative[lit].mean()),
        "rel_p99": float(np.percentile(relative[lit], 99)),
    }
    for c, channel in enumerate("rgb"):
        metrics[f"rel_mean_{channel}"] = float(relative[lit[:, c], c].mean())
    for name, mask in regions(reference[:, :4]).items():
        selected = lit & mask[:, None]
        assert selected.sum() > 100, name
        metrics[f"rel_mean_{name}"] = float(relative[selected].mean())
        metrics[f"rel_p99_{name}"] = float(np.percentile(relative[selected], 99))
    return metrics
