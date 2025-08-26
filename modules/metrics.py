import numpy as np
from scipy.ndimage import maximum_filter1d
from sklearn.metrics import precision_score
from typing import Optional, Tuple, Sequence, Union


def FDR(y_true: Sequence[int], y_pred: Sequence[int]) -> Tuple[float, int, int]:
    """
    False Discovery Rate (point-wise).
    Returns (FDR, FP count, TP count).
    """
    # force boolean
    y_true = np.asarray(y_true, bool)
    y_pred = np.asarray(y_pred, bool)
    tp = np.sum(y_true & y_pred)
    fp = np.sum(~y_true & y_pred)
    fdr = fp / (fp + tp) if (fp + tp) > 0 else 0.0
    return fdr, fp, tp


def F1_PA(
    y_true: Sequence[int], y_pred: Sequence[int], K: Optional[float] = None
) -> Tuple[float, float, float, float]:
    """
    Point‐Adjusted F1 (PA%K).
    - If K is None or K == 0: any hit in a true-event segment makes all points in that segment positive (PA).
    - If 0 < K < 1: segment is positive if proportion of positives ≥ K.
    - If K == 1: no adjustment (falls back to point‐wise predictions).
    Returns (F1_pa, precision_pa, recall_pa, FDR_pa).
    """
    import numpy as np

    y_true = np.asarray(y_true, int)
    y_pred = np.asarray(y_pred, int)

    # If K == 1, skip PA and use raw predictions
    if K is not None and K >= 1.0:
        y_pa = y_pred
    else:
        # detect event segments
        edges = np.diff(np.concatenate(([0], y_true, [0])))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0] - 1
        y_pa = y_pred.copy()
        for s, e in zip(starts, ends):
            seg = y_pred[s : e + 1]
            if K is None or K == 0.0:
                hit = seg.any()
            else:
                hit = seg.mean() >= K
            if hit:
                y_pa[s : e + 1] = 1

    # point‐wise counts
    TP_pa = int(np.sum((y_true == 1) & (y_pa == 1)))
    FP_pa = int(np.sum((y_true == 0) & (y_pa == 1)))
    FN_pa = int(np.sum((y_true == 1) & (y_pa == 0)))

    prec_pa = TP_pa / (TP_pa + FP_pa) if (TP_pa + FP_pa) > 0 else 0.0
    rec_pa = TP_pa / (TP_pa + FN_pa) if (TP_pa + FN_pa) > 0 else 0.0
    f1_pa = 2 * prec_pa * rec_pa / (prec_pa + rec_pa) if (prec_pa + rec_pa) > 0 else 0.0
    fdr_pa = FP_pa / (FP_pa + TP_pa) if (FP_pa + TP_pa) > 0 else 0.0

    return f1_pa, prec_pa, rec_pa, fdr_pa


def F1_W(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    recall_window_size: int = 100,
    precision_window_size: Optional[int] = None,
    bias: int = 0,
    *,
    timestamps: Optional[Sequence[Union[np.datetime64, float]]] = None,
    subject_ids: Optional[Sequence[int]] = None,
) -> Tuple[float, float, float, float]:
    """
    Calculates a dual-window, time-aware F1 score.

    This metric uses two separate windows to provide a nuanced evaluation:
    - `recall_window_size`: A wider, more lenient window to determine if a true
      event was detected at all (for calculating recall).
    - `precision_window_size`: A narrower, stricter window to determine if a
      prediction was temporally precise (for calculating precision).

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.
        recall_window_size: The size of the window (in samples or seconds)
            for calculating recall.
        precision_window_size: The size of the stricter window for calculating
            precision. If None, defaults to `recall_window_size`.
        bias: An offset (in samples or seconds) to shift the window.
            A positive bias shifts the window forward (y_true is expected
            after y_pred), a negative bias shifts it backward.
        timestamps: Optional sequence of timestamps for time-aware mode.
        subject_ids: Optional sequence of subject IDs for grouped processing.

    Returns:
        A tuple containing: (F1_score, Precision, Recall, FDR).
    """
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)

    if precision_window_size is None:
        precision_window_size = recall_window_size
    if precision_window_size > recall_window_size:
        raise ValueError(
            "precision_window_size cannot be larger than recall_window_size."
        )
    recall_half = recall_window_size // 2
    if abs(bias) > recall_half:
        raise ValueError(f"Bias must be between {-recall_half} and {recall_half}.")

    # 1) Choose mode: fixed-window (no timestamps) or time-aware
    if timestamps is None:
        # --- Fixed-window mode ---
        recall_half = recall_window_size // 2
        precision_half = precision_window_size // 2

        # For each point, is there a true event in its recall/precision window?
        # The 'origin' parameter handles the bias.
        has_event_in_recall_window = (
            maximum_filter1d(y_true.astype(int), size=2 * recall_half + 1, origin=bias)
            > 0
        )
        has_event_in_precision_window = (
            maximum_filter1d(
                y_true.astype(int), size=2 * precision_half + 1, origin=bias
            )
            > 0
        )

        # For each true event, is there a prediction in its recall window?
        has_pred_in_recall_window = (
            maximum_filter1d(y_pred.astype(int), size=2 * recall_half + 1, origin=-bias)
            > 0
        )

        # TP for Precision: A prediction that aligns with a true event in the *stricter* window.
        TP_p = np.sum(y_pred & has_event_in_precision_window)
        # FP: A prediction that has NO true event even in the *lenient* recall window.
        FP = np.sum(y_pred & ~has_event_in_recall_window)

        # TP for Recall: A true event that has a prediction in its *lenient* recall window.
        TP_r = np.sum(y_true & has_pred_in_recall_window)
        # FN: A true event that has NO prediction even in its *lenient* recall window.
        FN = np.sum(y_true & ~has_pred_in_recall_window)

    else:
        # --- Time-aware mode ---
        ts = np.asarray(timestamps)
        if np.issubdtype(ts.dtype, np.datetime64):
            ts = ((ts - ts.min()) / np.timedelta64(1, "s")).astype(float)
        else:
            ts = ts.astype(float)
            ts -= ts.min()

        # Overlap checks
        if subject_ids is None:
            if np.unique(ts).size != ts.size:
                raise ValueError(
                    "Overlapping timestamps detected but no subject_ids provided."
                )
        else:
            sarr = np.asarray(subject_ids)
            for sid in np.unique(sarr):
                mask = sarr == sid
                if np.unique(ts[mask]).size != mask.sum():
                    raise ValueError(f"Overlapping timestamps for subject {sid}.")

        def _process_dual_window_vectorized(tseq, y_true_bin, y_pred_bin):
            """Vectorized implementation for speed."""
            # Find indices of true events and predictions
            true_indices = np.where(y_true_bin)[0]
            pred_indices = np.where(y_pred_bin)[0]

            if len(pred_indices) == 0:
                FN = len(true_indices)
                return 0, 0, 0, FN

            if len(true_indices) == 0:
                FP = len(pred_indices)
                return 0, FP, 0, 0

            t_true = tseq[true_indices]
            t_pred = tseq[pred_indices]

            recall_half = recall_window_size / 2.0
            precision_half = precision_window_size / 2.0

            # --- Calculate TP_p and FP (based on predictions) ---
            t_center_pred = t_pred + bias

            # Find recall window boundaries for all predictions
            recall_starts = np.searchsorted(
                t_true, t_center_pred - recall_half, side="left"
            )
            recall_ends = np.searchsorted(
                t_true, t_center_pred + recall_half, side="right"
            )

            # Find precision window boundaries for all predictions
            precision_starts = np.searchsorted(
                t_true, t_center_pred - precision_half, side="left"
            )
            precision_ends = np.searchsorted(
                t_true, t_center_pred + precision_half, side="right"
            )

            # A prediction is a TP_p if it has a true event in its precision window
            has_event_in_precision_window = (precision_ends - precision_starts) > 0
            TP_p = np.sum(has_event_in_precision_window)

            # A prediction is a FP if it has NO true event in its recall window
            has_event_in_recall_window = (recall_ends - recall_starts) > 0
            FP = np.sum(~has_event_in_recall_window)

            # --- Calculate FN (based on true events) ---
            t_center_true = t_true - bias

            # Find recall window boundaries for all true events
            recall_starts_for_true = np.searchsorted(
                t_pred, t_center_true - recall_half, side="left"
            )
            recall_ends_for_true = np.searchsorted(
                t_pred, t_center_true + recall_half, side="right"
            )

            # A true event is a FN if it has NO prediction in its recall window
            has_pred_in_recall_window = (
                recall_ends_for_true - recall_starts_for_true
            ) > 0
            FN = np.sum(~has_pred_in_recall_window)

            # TP for recall is the total number of true events minus the misses (FN)
            TP_r = len(true_indices) - FN

            return TP_p, FP, TP_r, FN

        TP_p = FP = TP_r = FN = 0
        if subject_ids is None or len(np.unique(subject_ids)) == 1:
            TP_p, FP, TP_r, FN = _process_dual_window_vectorized(ts, y_true, y_pred)
        else:
            from joblib import Parallel, delayed

            sub_arr = np.asarray(subject_ids)
            unique_sids = np.unique(sub_arr)

            # Create a list of jobs to run in parallel
            tasks = [
                delayed(_process_dual_window_vectorized)(
                    ts[sub_arr == sid], y_true[sub_arr == sid], y_pred[sub_arr == sid]
                )
                for sid in unique_sids
            ]

            # Run the jobs in parallel (n_jobs=-1 uses all available cores)
            results = Parallel(n_jobs=-1)(tasks)

            # Sum the results from all parallel jobs
            # results is a list of tuples like [(TP_p, FP, TP_r, FN), ...]
            TP_p, FP, TP_r, FN = np.sum(results, axis=0)

    # 3) Final dual-window metrics
    # Precision uses the stricter TP count (TP_p)
    precision = TP_p / (TP_p + FP) if (TP_p + FP) > 0 else 0.0
    # Recall uses the more lenient TP count (TP_r)
    recall = TP_r / (TP_r + FN) if (TP_r + FN) > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    fdr = FP / (FP + TP_p) if (FP + TP_p) > 0 else 0.0

    return f1, precision, recall, fdr
