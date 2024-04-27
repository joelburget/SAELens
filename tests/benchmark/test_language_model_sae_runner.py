import torch
import os

from sae_lens.training.config import LanguageModelSAERunnerConfig
from sae_lens.training.lm_runner import language_model_sae_runner

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB__SERVICE_WAIT"] = "300"


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")
    cfg = LanguageModelSAERunnerConfig(
        # Data Generating Function (Model + Training Distibuion)
        model_name="gelu-2l",
        hook_point="blocks.0.hook_mlp_out",
        hook_point_layer=0,
        d_in=512,
        dataset_path="NeelNanda/c4-tokenized-2b",
        is_dataset_tokenized=True,
        # SAE Parameters
        expansion_factor=16,
        b_dec_init_method="mean",  # not ideal but quicker when testing code.
        # Training Parameters
        lr=1e-4,
        l1_coefficient=3e-4,
        train_batch_size=4096,
        context_size=128,
        # Activation Store Parameters
        n_batches_in_buffer=24,
        training_tokens=1_000_000 * 10,
        store_batch_size=32,
        # Resampling protocol
        use_ghost_grads=True,
        feature_sampling_window=3000,  # in steps
        dead_feature_window=5000,
        dead_feature_threshold=1e-8,
        # WANDB
        log_to_wandb=True,
        wandb_project="polar",
        wandb_entity=None,
        wandb_log_frequency=100,
        # Misc
        device=device,
        seed=42,
        n_checkpoints=5,
        checkpoint_path="checkpoints",
        dtype=torch.float32,
    )

    print("training")
    sparse_autoencoder = language_model_sae_runner(cfg)

    assert sparse_autoencoder is not None
    # know whether or not this works by looking at the dashbaord!
