import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset
from tqdm import tqdm


class CovariateMapDataset(Dataset):
    """Lazily generate context+horizon windows on demand with padding."""

    def __init__(
        self,
        df,
        context_len,
        horizon_len,
        covariate_cols=["bvp", "eda", "temp", "hr", "acc_x", "acc_y", "acc_z"],
        target_col="Event_Occured",
        time_col="Time",
        subject_col="subject_id",
        freq="250ms",
        means=None,
        stds=None,
    ):
        self.covariates = covariate_cols
        self.target_col = target_col
        self.freq_td = pd.to_timedelta(freq)

        # parse context and horizon into points + timedeltas
        def _parse_length(L):
            if isinstance(L, int):
                return L, L * self.freq_td
            td = pd.to_timedelta(L)
            return int(td / self.freq_td), td

        self.context_pts, self.context_td = _parse_length(context_len)
        self.horizon_pts, self.horizon_td = _parse_length(horizon_len)
        self.window_pts = self.context_pts + self.horizon_pts
        self.window_td = self.context_td + self.horizon_td
        self.step_td = min(self.context_td, self.horizon_td)

        # prepare global normalization stats
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values([subject_col, time_col])
        cov_df = df[self.covariates].astype(np.float32)
        if means is not None and stds is not None:
            self.means = means
            self.stds = stds
        else:
            # compute means and stds, replacing 0 stds with a small value
            # to avoid division by zero in normalization
            self.means = cov_df.mean()
            self.stds = cov_df.std().replace(0, 1e-6)

        # split by subject for fast slicing
        self.by_subject = {
            sid: grp.set_index(time_col).sort_index()
            for sid, grp in df.groupby(subject_col)
        }

        # build index of all (subject, t) windows
        self.index = []
        for sid, grp in tqdm(self.by_subject.items(), desc="Index windows"):
            start_t = grp.index.min() + self.horizon_td
            end_t = grp.index.max() - self.window_td
            t = start_t
            while t <= end_t:
                self.index.append((sid, t))
                t += self.step_td

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        sid, t = self.index[idx]
        grp = self.by_subject[sid]

        # target window: [t, t + window_td]
        start_t, stop_t = t, t + self.window_td
        idx_tgt = pd.date_range(start_t, periods=self.window_pts, freq=self.freq_td)
        padded_tgt = grp.loc[start_t:stop_t].reindex(idx_tgt)
        tgt_np = padded_tgt[self.target_col].fillna(0).to_numpy(np.float32)

        # covariates window shifted back by horizon: [t - horizon_td, t + context_td]
        start_cov = t - self.horizon_td
        stop_cov = t + self.context_td
        idx_cov = pd.date_range(start_cov, periods=self.window_pts, freq=self.freq_td)
        padded_cov = grp.loc[start_cov:stop_cov].reindex(idx_cov)

        cov_norm = {}
        for c in self.covariates:
            raw = padded_cov[c].fillna(0).to_numpy(np.float32)
            cov_norm[c] = raw  # (raw - self.means[c]) / self.stds[c]

        inp = tgt_np[: self.context_pts]
        out = tgt_np[self.context_pts :]

        return {
            "inputs": torch.from_numpy(inp) * 0,
            "outputs": np.array(out),
            **{c: torch.from_numpy(cov_norm[c]) for c in self.covariates},
            "subject_id": np.repeat(int(sid), out.shape[0]),
            "timestamps": np.array(idx_tgt[self.context_pts :], dtype="datetime64[ns]"),
        }
