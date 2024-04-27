import torch

# from sae_lens.training.session_loader import LMSparseAutoencoderSessionloader
from sae_lens.analysis.dashboard_runner import DashboardRunner

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    sae_path = "checkpoints/klnkkga3/final/gelu-2l_blocks.0.hook_mlp_out_8192_/"
    # model, sparse_autoencoder, activation_store = (
    #     LMSparseAutoencoderSessionloader.load_pretrained_sae(path=sae_path)
    # )

    runner = DashboardRunner(
        sae_path=sae_path,
        wandb_artifact_path="joelb/polar/dashboard",
        use_wandb=True,
    )
    runner.run()
