"""Multiclass variational QNN and genuine hierarchical quantum CNN."""

import math

import pennylane as qml
import torch
from torch import nn

from qaml_ctf.config import PilotConfig
from qaml_ctf.data.preprocessing import ImagePreprocessor


def encode(angles, qubits: int, encoding: str):
    if encoding == "amplitude":
        qml.StatePrep(angles, wires=range(qubits), normalize=False)
        return
    for wire in range(qubits):
        qml.RY(angles[..., wire], wires=wire)
        if encoding == "ry_rz":
            qml.RZ(angles[..., wire], wires=wire)


def convolution(theta, pair):
    """A shared trainable two-qubit unitary applied to neighboring wires."""
    a, b = pair
    qml.RY(theta[0], wires=a)
    qml.RY(theta[1], wires=b)
    qml.RZ(theta[2], wires=a)
    qml.RZ(theta[3], wires=b)
    qml.IsingXX(theta[4], wires=pair)
    qml.IsingYY(theta[5], wires=pair)
    qml.IsingZZ(theta[6], wires=pair)


def pooling(theta, discard: int, retain: int):
    """Coherent conditional pooling; discard is never used in subsequent stages.

    Applying a controlled unitary then ignoring the control implements reduction
    by partial trace, equivalent to unobserved measurement and conditional action.
    It does not inject noise or approximate pooling with classical features.
    """
    qml.RY(theta[0], wires=discard)
    qml.CRY(theta[1], wires=[discard, retain])
    qml.PauliX(wires=discard)
    qml.CRY(theta[2], wires=[discard, retain])
    qml.PauliX(wires=discard)


class QuantumClassifier(nn.Module):
    def __init__(
        self,
        config: PilotConfig,
        image_shape: tuple[int, ...],
        family: str,
        preprocessor: nn.Module | None = None,
    ):
        super().__init__()
        if family not in ("qnn", "qcnn"):
            raise ValueError("Unknown quantum model family")
        self.config = config
        self.family = family
        if preprocessor is None and config.encoding == "amplitude":
            raise ValueError(
                "Amplitude encoding requires an explicit fitted amplitude preprocessor"
            )
        self.preprocess = (
            preprocessor
            if preprocessor is not None
            else ImagePreprocessor(image_shape, config.qubits, config.preprocessing)
        )
        n = config.qubits
        stages = math.ceil(math.log2(n))
        if family == "qnn":
            self.quantum_weights = nn.Parameter(
                0.2 * torch.randn(config.layers, n, 3, dtype=torch.float64)
            )
            features = 2 * n
        else:
            self.conv_weights = nn.Parameter(
                0.2 * torch.randn(stages, config.layers, 7, dtype=torch.float64)
            )
            self.pool_weights = nn.Parameter(
                0.2 * torch.randn(stages, 3, dtype=torch.float64)
            )
            features = 3
        self.head = nn.Linear(features, len(config.classes), dtype=torch.float64)
        device = qml.device("default.qubit", wires=n, shots=None)

        def circuit(angles, weights, pool_weights=None):
            encode(angles, n, config.encoding)
            if family == "qnn":
                for layer in range(config.layers):
                    for wire in range(n):
                        qml.Rot(*weights[layer, wire], wires=wire)
                    for wire in range(n):
                        qml.CNOT(wires=[wire, (wire + 1) % n])
                return tuple(qml.expval(qml.PauliZ(w)) for w in range(n)) + tuple(
                    qml.expval(qml.PauliZ(w) @ qml.PauliZ((w + 1) % n))
                    for w in range(n)
                )
            active = list(range(n))
            for stage in range(stages):
                for layer in range(config.layers):
                    # Shared convolution on even bonds, then shifted bonds.
                    for offset in (0, 1):
                        for k in range(offset, len(active) - 1, 2):
                            convolution(
                                weights[stage, layer], [active[k], active[k + 1]]
                            )
                for k in range(0, len(active) - 1, 2):
                    pooling(pool_weights[stage], active[k], active[k + 1])
                # Carry an unmatched wire into the next scale (10/12-qubit support).
                active = active[1::2] + ([active[-1]] if len(active) % 2 else [])
            return tuple(
                qml.expval(op(active[0])) for op in (qml.PauliX, qml.PauliY, qml.PauliZ)
            )

        self.qnode = qml.QNode(
            circuit, device, interface="torch", diff_method="backprop"
        )

    def quantum_features(self, images):
        angles = self.preprocess(images)
        values = (
            self.qnode(angles, self.quantum_weights)
            if self.family == "qnn"
            else self.qnode(angles, self.conv_weights, self.pool_weights)
        )
        return torch.stack(values, dim=-1)

    def forward(self, images):
        return self.head(self.quantum_features(images))

    def predict_proba(self, images):
        # Keep gradients for white-box callers; inference callers use no_grad.
        return self(images).softmax(dim=-1)
