import os


def determine_compute_resources(per_core_batch_size, backend="gpu", max_workers=6):
    """Return (num_devices, dataloader_batch_size, num_workers)."""
    import jax

    try:
        num_devices = jax.local_device_count() if backend == "gpu" else 1
        if num_devices == 0:
            print("No GPUs found; falling back to CPU.")
            num_devices = 1
    except:
        print("Could not query devices; defaulting to 1.")
        num_devices = 1

    batch_size = (
        per_core_batch_size * num_devices if num_devices > 1 else per_core_batch_size
    )
    print(f"Using {num_devices} device(s); DataLoader batch size = {batch_size}.")
    cpus = os.cpu_count() or 1
    workers = min(max_workers, cpus)
    print(f"Using {workers} DataLoader worker(s).")
    return num_devices, batch_size, workers


def initialize_timesfm(context_len, horizon_len, per_core_batch_size, backend="gpu"):
    import timesfm

    """Instantiate and return a TimesFM model with its hparams."""
    tfm_hparams = timesfm.TimesFmHparams(
        backend=backend,
        per_core_batch_size=per_core_batch_size,
        context_len=context_len,
        horizon_len=horizon_len,
        num_layers=50,
        use_positional_embedding=False,
    )
    print(
        f"Initializing TimesFM (backend={backend}, per_core_batch_size={per_core_batch_size})"
    )
    return timesfm.TimesFm(
        hparams=tfm_hparams,
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
        ),
    )
