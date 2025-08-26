import os

# 1) Must come before any JAX / TF import:
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PMAP_USE_TENSORSTORE"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # hide TF INFO/WARNING
os.environ["JAX_LOG_COMPILES"] = "0"

import click
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp
import time

from modules.data_split import subject_level_split, temporal_within_subject_split
from modules.compute_resources import determine_compute_resources, initialize_timesfm
from modules.dataset import CovariateMapDataset, custom_collate
from modules.forecast import forecast_multivariate


@click.command()
@click.argument("data_set", type=str)
@click.argument("context_len", type=int)
@click.argument("horizon_len", type=int)
@click.argument("label_adjustment", default="", type=str)
@click.argument("batch_size_per_core", default=2400, type=int)
@click.option("--val", is_flag=True, help="Run in val mode.")
@click.option("--test", is_flag=True, help="Run in test mode.")
def main(
    data_set, context_len, horizon_len, label_adjustment, batch_size_per_core, val, test
):
    # Suffix for label adjustment if provided
    ELA_suffix = f"_{label_adjustment}" if label_adjustment else ""
    # Load dataset CSV
    df = pd.read_csv(
        os.path.join(data_set, f"All{ELA_suffix}.csv"), parse_dates=["Time"]
    )

    if data_set == "ROAD":
        # Use subject-level split for ROAD dataset
        val_df, test_df = subject_level_split(df, test_frac=0.8)
    else:
        # Use temporal split for other datasets
        val_df, test_df = temporal_within_subject_split(df, test_frac=0.8)
        print(f"Using temporal split for {data_set} dataset.")
        print(f"Validation set size: {len(val_df)} samples")
        print(f"Test set size: {len(test_df)} samples")

    # Determine compute resources (devices, batch size, workers)
    num_devices, actual_batch_size, num_workers = determine_compute_resources(
        per_core_batch_size=batch_size_per_core,
        backend="gpu",
    )

    # Initialize the TimesFM model
    tfm = initialize_timesfm(
        context_len, horizon_len, batch_size_per_core, backend="gpu"
    )

    # Helper function to create a DataLoader for a dataset
    def make_loader(ds):
        return DataLoader(
            ds,
            batch_size=actual_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            multiprocessing_context=mp.get_context("spawn"),
            collate_fn=custom_collate,
        )

    # Prepare validation dataset
    val_ds = CovariateMapDataset(val_df, context_len, horizon_len)
    # Run inference & save for validation set if requested
    if val:
        start_time = time.time()
        forecast_multivariate(
            make_loader(val_ds),
            tfm,
            data_set,
            context_len,
            horizon_len,
            ELA_suffix=ELA_suffix,
            data_set_flag="val",
        )
        end_time = time.time()
        elapsed_minutes = (end_time - start_time) / 60

        print(f"Prediction step for Validation took {elapsed_minutes:.2f} minutes.")

    # Run inference & save for test set if requested
    if test:
        start_time = time.time()
        # Use means and stds from validation set for normalization
        test_ds = CovariateMapDataset(
            test_df, context_len, horizon_len, means=val_ds.means, stds=val_ds.stds
        )
        del val_ds  # Free memory
        forecast_multivariate(
            make_loader(test_ds),
            tfm,
            data_set,
            context_len,
            horizon_len,
            ELA_suffix=ELA_suffix,
            data_set_flag="test",
        )
        end_time = time.time()
        elapsed_minutes = (end_time - start_time) / 60

        print(f"Prediction step for Test took {elapsed_minutes:.2f} minutes.")


if __name__ == "__main__":
    try:
        # Set multiprocessing start method to 'spawn' for compatibility
        mp.set_start_method("spawn", force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        # Might have already been set or not applicable in some environments
        print(
            "Could not set multiprocessing start method to 'spawn'. It might already be set or this is not the main module."
        )
        pass
    main()
