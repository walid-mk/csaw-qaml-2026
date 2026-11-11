"""Extend native QASM layers without changing encoding, pairs or readout."""

import pennylane as qml
import torch
from torch import nn

from qaml_ctf.models import encode
from qaml_ctf.road_architecture.config import Architecture
from qaml_ctf.road_architecture.models import ArchitectureModel

PAIRS = ((1, 2), (3, 4), (5, 6))


def parameter_count(design, layers):
    if (
        design not in ("primary", "medium")
        or type(layers) is not int
        or layers not in (1, 2, 3, 4)
    ):
        raise ValueError("Expected Primary/Medium at 1–4 native layers")
    return 16 * layers if design == "primary" else 8 + 16 * (layers - 1)


def body(weights, design, layers):
    position = 0
    for layer in range(layers):
        if design == "primary":
            for wire in range(8):
                qml.RY(weights[position], wires=wire)
                qml.RZ(weights[position + 1], wires=wire)
                position += 2
        else:
            if layer:
                for wire in range(8):
                    qml.RX(weights[position], wires=wire)
                    position += 1
            for wire in range(8):
                qml.RY(weights[position], wires=wire)
                position += 1
        for pair in PAIRS:
            qml.CZ(wires=pair)


class DepthModel(ArchitectureModel):
    """Reuse the frozen hybrid model's head, forward, hash and resource methods."""

    def __init__(self, design, layers, classes, preprocessor):
        nn.Module.__init__(self)
        count = parameter_count(design, layers)
        if type(classes) is not int or classes not in (3, 4):
            raise ValueError("Road Signs contract has three or four classes")
        self.design, self.layers = design, layers
        self.architecture = Architecture(f"{design}_l{layers}", "qnn")
        self.classes, self.head_kind, self.probability_floor = classes, "linear", 1e-12
        self.preprocess = preprocessor
        self.quantum_weights = nn.Parameter(
            0.2 * torch.randn(count, dtype=torch.float64)
        )
        self.head = nn.Linear(16, classes, dtype=torch.float64)

        def circuit(angles, weights):
            encode(angles, 8, "ry")
            body(weights, design, layers)
            return tuple(qml.expval(qml.PauliZ(w)) for w in range(8)) + tuple(
                qml.expval(qml.PauliZ(w) @ qml.PauliZ((w + 1) % 8)) for w in range(8)
            )

        self.qnode = qml.QNode(
            circuit,
            qml.device("default.qubit", wires=8, shots=None),
            interface="torch",
            diff_method="backprop",
        )
