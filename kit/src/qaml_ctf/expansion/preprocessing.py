"""Live differentiable, train-only scaling/PCA and classical amplitude mapping."""

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F


class ExpansionPreprocessor(nn.Module):
    def __init__(self, input_shape, components: int, qubits: int, encoding: str):
        super().__init__()
        self.input_shape = tuple(input_shape)
        self.qubits = qubits
        self.encoding = encoding
        pixels = int(np.prod(input_shape))
        for key, value in {
            "input_center": torch.zeros(pixels, dtype=torch.float64),
            "input_scale": torch.ones(pixels, dtype=torch.float64),
            "pca_center": torch.zeros(pixels, dtype=torch.float64),
            "components": torch.zeros(components, pixels, dtype=torch.float64),
            "angle_scale": torch.ones(components, dtype=torch.float64),
            "explained_variance_ratio": torch.zeros(components, dtype=torch.float64),
            "fitted": torch.tensor(False),
        }.items():
            self.register_buffer(key, value)

    def fit(self, training: torch.Tensor, *, standardize: bool, seed: int):
        flat = (
            training.detach()
            .cpu()
            .numpy()
            .reshape(len(training), -1)
            .astype(np.float64)
        )
        if standardize:
            scaler = StandardScaler().fit(flat)
            self.input_center.copy_(torch.from_numpy(scaler.mean_))
            self.input_scale.copy_(torch.from_numpy(scaler.scale_))
            flat = scaler.transform(flat)
        pca = PCA(
            n_components=len(self.components),
            whiten=False,
            svd_solver="randomized",
            random_state=seed,
        ).fit(flat)
        projected = pca.transform(flat)
        self.pca_center.copy_(torch.from_numpy(pca.mean_))
        self.components.copy_(torch.from_numpy(pca.components_))
        self.angle_scale.copy_(
            torch.from_numpy(np.maximum(projected.std(axis=0), 1e-8))
        )
        self.explained_variance_ratio.copy_(
            torch.from_numpy(pca.explained_variance_ratio_)
        )
        self.fitted.fill_(True)
        return self

    def project(self, inputs):
        if not bool(self.fitted) or tuple(inputs.shape[1:]) != self.input_shape:
            raise ValueError("Preprocessor not fitted or input shape changed")
        flat = inputs.to(torch.float64).flatten(1)
        standardized = (flat - self.input_center) / self.input_scale
        return (standardized - self.pca_center) @ self.components.T

    def forward(self, inputs):
        projected = self.project(inputs)
        if self.encoding == "ry":
            return torch.atan(
                projected[:, : self.qubits] / self.angle_scale[: self.qubits]
            )
        if self.encoding != "amplitude":
            raise ValueError("Unknown encoding")
        keep = min(projected.shape[1], 2**self.qubits)
        vector = F.pad(projected[:, :keep], (0, 2**self.qubits - keep))
        norm = vector.norm(dim=1, keepdim=True)
        fallback = torch.zeros_like(vector)
        fallback[:, 0] = 1
        # Clamp before where: inactive branches must not produce NaN gradients.
        return torch.where(norm > 1e-12, vector / norm.clamp_min(1e-12), fallback)
