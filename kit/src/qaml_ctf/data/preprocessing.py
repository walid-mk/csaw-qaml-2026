"""Train-only preprocessing implemented in Torch to preserve image gradients."""

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn
from torch.nn import functional as F


class ImagePreprocessor(nn.Module):
    def __init__(self, image_shape: tuple[int, ...], qubits: int, method: str = "pca"):
        super().__init__()
        self.image_shape = tuple(image_shape)
        self.qubits = qubits
        self.method = method
        pixels = int(np.prod(image_shape))
        self.register_buffer("center", torch.zeros(pixels, dtype=torch.float64))
        self.register_buffer(
            "components", torch.zeros(qubits, pixels, dtype=torch.float64)
        )
        self.register_buffer("scale", torch.ones(qubits, dtype=torch.float64))
        self.register_buffer("fitted", torch.tensor(False))

    def fit(self, training_images: torch.Tensor):
        if self.method == "pca":
            flat = (
                training_images.detach().cpu().numpy().reshape(len(training_images), -1)
            )
            pca = PCA(n_components=self.qubits, svd_solver="full").fit(flat)
            self.center.copy_(torch.from_numpy(pca.mean_))
            self.components.copy_(torch.from_numpy(pca.components_))
            self.scale.copy_(
                torch.from_numpy(np.maximum(np.sqrt(pca.explained_variance_), 1e-8))
            )
        elif self.method != "resize":
            raise ValueError("Unknown preprocessing method")
        self.fitted.fill_(True)
        return self

    def forward(self, images: torch.Tensor):
        if not bool(self.fitted):
            raise ValueError("Preprocessor must be fitted on training images")
        if images.ndim != 4 or tuple(images.shape[1:]) != self.image_shape:
            raise ValueError(f"Expected NCHW images with shape {self.image_shape}")
        images = images.to(dtype=torch.float64)
        if self.method == "pca":
            projected = (images.flatten(1) - self.center) @ self.components.T
            return torch.atan(projected / self.scale)
        resized = F.adaptive_avg_pool2d(
            images.mean(dim=1, keepdim=True), (2, self.qubits // 2)
        )
        return (resized.flatten(1) - 0.5) * torch.pi
