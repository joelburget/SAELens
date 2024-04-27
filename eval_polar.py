import torch

# from sae_lens.training.evals import run_evals
from sae_lens.training.session_loader import LMSparseAutoencoderSessionloader

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # layer = 8 # pick a layer you want.
    # REPO_ID = "jbloom/GPT2-Small-SAEs"
    # FILENAME = f"final_sparse_autoencoder_gpt2-small_blocks.{layer}.hook_resid_pre_24576.pt"
    # path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    model, sparse_autoencoder, activation_store = (
        LMSparseAutoencoderSessionloader.load_pretrained_sae(
            path="checkpoints/klnkkga3/final/gelu-2l_blocks.0.hook_mlp_out_8192_/"
        )
    )
    sparse_autoencoder.eval()
