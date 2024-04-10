import os
from pathlib import Path

import torch

from sae_lens.training.sae_group import SparseAutoencoderDictionary
from tests.unit.helpers import build_sae_cfg


def test_SparseAutoencoderDictionary_initializes_all_permutations_of_list_params():
    cfg = build_sae_cfg(
        d_in=5,
        lr=[0.01, 0.001],
        expansion_factor=[2, 4],
    )
    sae_group = SparseAutoencoderDictionary(cfg)
    assert len(sae_group) == 4
    lr_sae_combos = [(ae.cfg.lr, ae.cfg.d_sae) for _, ae in sae_group]
    assert (0.01, 10) in lr_sae_combos
    assert (0.01, 20) in lr_sae_combos
    assert (0.001, 10) in lr_sae_combos
    assert (0.001, 20) in lr_sae_combos


def test_SparseAutoencoderDictionary_replaces_layer_with_actual_layer():
    cfg = build_sae_cfg(
        hook_point="blocks.{layer}.attn.hook_q",
        hook_point_layer=5,
    )
    sae_group = SparseAutoencoderDictionary(cfg)
    assert len(sae_group) == 1
    sae = next(iter(sae_group))[1]
    assert sae.cfg.hook_point == "blocks.5.attn.hook_q"


def test_SparseAutoencoderDictionary_train_and_eval():
    cfg = build_sae_cfg(
        lr=[0.01, 0.001],
        expansion_factor=[2, 4],
    )
    sae_group = SparseAutoencoderDictionary(cfg)
    sae_group.train()
    for _, sae in sae_group:
        assert sae.training is True
    sae_group.eval()
    for _, sae in sae_group:
        assert sae.training is False
    sae_group.train()
    for _, sae in sae_group:
        assert sae.training is True


def test_SparseAutoencoderDictionary_save_and_load_model(tmp_path: Path) -> None:

    cfg = build_sae_cfg(
        lr=[0.01, 0.001],
        expansion_factor=[2, 4],
    )
    sae_group = SparseAutoencoderDictionary(cfg)
    sae_group.save_saes(f"{tmp_path}/sae_group_saving")

    # Check that the files were saved

    # 2 subfolders in the main folder
    path = f"{tmp_path}/sae_group_saving"
    assert len(os.listdir(path)) == 4
    # each should
    for subfolder in os.listdir(path):
        contents = os.listdir(f"{path}/{subfolder}")
        print(contents)
        assert len(contents) == 2
        assert "cfg.json" in contents
        assert "sae_weights.safetensors" in contents

    # Load the model
    new_sae_group = SparseAutoencoderDictionary.load_from_pretrained(path)

    assert set(sae_group.autoencoders.keys()) == set(new_sae_group.autoencoders.keys())
    for key in sae_group.autoencoders.keys():

        # check that the model weights match
        for weight_key in sae_group[key].state_dict().keys():
            assert torch.allclose(
                sae_group[key].state_dict()[weight_key],
                new_sae_group[key].state_dict()[weight_key],
            )
        # check that the config matches
        sae_group[key].cfg.verbose = new_sae_group[key].cfg.verbose = False
        sae_group[key].cfg.checkpoint_path = new_sae_group[key].cfg.checkpoint_path
        assert sae_group[key].cfg == new_sae_group[key].cfg
