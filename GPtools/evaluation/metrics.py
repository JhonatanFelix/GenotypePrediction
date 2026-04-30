"""regression metrics"""

import numpy as np

def _as_float_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return y_true, y_pred

def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient r**2"""

    y_true, y_pred = _as_float_arrays(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if np.isclose(ss_tot, 0.0):
        return 0.0
    return float(1.0 - ss_res / ss_tot)

def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation Coefficient"""

    y_true, y_pred = _as_float_arrays(y_true, y_pred)
    y_true_centered = y_true - np.mean(y_true)
    y_pred_centered = y_pred - np.mean(y_pred)
    denominator = np.sqrt(np.sum(y_true_centered ** 2) * np.sum(y_pred_centered ** 2))
    if np.isclose(denominator, 0.0):
        return 0.0
    return float(np.sum(y_true_centered * y_pred_centered) / denominator)
