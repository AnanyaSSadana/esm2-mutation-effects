"""Shared loading and device helpers for fair-esm."""

from __future__ import annotations

MODEL_NAME = "esm2_t6_8M_UR50D"
REPRESENTATION_LAYER = 6


def choose_device(requested: str):
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(device):
    import esm

    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model.eval()
    model = model.to(device)
    return model, alphabet
