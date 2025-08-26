import pandas as pd


def subject_level_split(df, test_frac=0.8):
    """Split by subjects preserving temporal integrity."""
    subjects = df["subject_id"].unique()
    n_test = int(len(subjects) * test_frac)
    subject_stats = (
        df.groupby("subject_id")["Event_Occured"]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values("mean")
    )
    step = len(subjects) // n_test if n_test > 0 else len(subjects)
    test_subjects = subject_stats.iloc[::step]["subject_id"].head(n_test).values
    test_df = df[df["subject_id"].isin(test_subjects)]
    val_df = df[~df["subject_id"].isin(test_subjects)]
    return val_df, test_df


def temporal_within_subject_split(df, test_frac=0.8):
    """Split last test_frac of each subject’s timeline."""
    val_parts, test_parts = [], []
    for sid, grp in df.groupby("subject_id"):
        grp_sorted = grp.sort_values("Time")
        n_test = max(1, int(len(grp_sorted) * test_frac))
        test_parts.append(grp_sorted.iloc[-n_test:])
        val_parts.append(grp_sorted.iloc[:-n_test])
    return pd.concat(val_parts, ignore_index=True), pd.concat(
        test_parts, ignore_index=True
    )
