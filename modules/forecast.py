import os, numpy as np, pandas as pd
from tqdm import tqdm


def forecast_multivariate(
    dataloader,
    tfm,
    output_directory,
    inputsize,
    predlength,
    covariate_cols=["bvp", "eda", "temp", "hr", "acc_x", "acc_y", "acc_z"],
    ELA_suffix="",
    data_set_flag="val",
):
    """Run model on each batch, collect and save predictions+truth+timestamps."""

    all_mv, all_true, all_time, all_subj = [], [], [], []
    for batch in tqdm(dataloader, desc="Forecast batches"):
        B = batch["inputs"].size(0)
        pt, _ = tfm.forecast_with_covariates(
            inputs=batch["inputs"],
            dynamic_numerical_covariates={c: batch[c] for c in covariate_cols},
            dynamic_categorical_covariates={},
            static_numerical_covariates={},
            static_categorical_covariates={},
            freq=[0] * B,
            normalize_xreg_target_per_input=False,
        )

        all_mv.append(pt)
        all_true.append(batch["outputs"])
        all_time.append(batch["timestamps"])
        all_subj.append(batch["subject_id"])

    P_mv = np.concatenate(all_mv, axis=0)
    T = np.concatenate(all_true, axis=0).astype(int)
    time = np.concatenate(all_time, axis=0)
    subj = np.concatenate(all_subj, axis=0)

    base = f"tfm_C{inputsize}_H{predlength}{ELA_suffix}_{data_set_flag}"
    np.save(os.path.join(output_directory, f"2{base}_pred.npy"), P_mv)
    np.save(os.path.join(output_directory, f"2{base}_true.npy"), T)
    np.save(os.path.join(output_directory, f"2{base}_timestamps.npy"), time)
    np.save(os.path.join(output_directory, f"2{base}_subject_ids.npy"), subj)
