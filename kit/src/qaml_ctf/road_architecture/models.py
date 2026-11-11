"""Live PennyLane circuits, exact restricted QASM translation and Born readout."""

import re
from collections import Counter
from pathlib import Path

import pennylane as qml
import torch
from torch import nn

from qaml_ctf.models import convolution, encode, pooling

QASM_FILES = {
    "qnn_qasm_medium": "ansatz_8q_med_expr_low_ent.qasm",
    "qnn_qasm_primary": "ansatz_8q_robust_primary.qasm",
}
QASM_ROOT = Path(__file__).resolve().parents[4] / "Ansatzes_Paper/8-Qubits"


def parse_qasm(path):
    """Reject unsupported syntax; retain every gate and theta index in source order.

    This is deliberately a translator for the supplied R{X,Y,Z}/CZ subset,
    not a general OpenQASM 3 interpreter. No source rewriting/execution occurs.
    """
    gates, declared, used, register = [], [], [], 0
    for original in Path(path).read_text().splitlines():
        line = original.split("//", 1)[0].strip()
        if not line or line in ("OPENQASM 3.0;", 'include "stdgates.inc";'):
            continue
        if line == "qubit[8] q;":
            register += 1
            continue
        if match := re.fullmatch(r"input float\[64\] theta_(\d+);", line):
            declared.append(int(match[1]))
            continue
        if match := re.fullmatch(r"(rx|ry|rz)\(theta_(\d+)\) q\[([0-7])\];", line):
            parameter, wire = int(match[2]), int(match[3])
            used.append(parameter)
            gates.append((match[1], parameter, (wire,)))
            continue
        if match := re.fullmatch(r"cz q\[([0-7])\], q\[([0-7])\];", line):
            wires = (int(match[1]), int(match[2]))
            if wires[0] == wires[1]:
                raise ValueError("CZ requires distinct wires")
            gates.append(("cz", None, wires))
            continue
        raise ValueError(f"Unsupported QASM statement: {line}")
    if (
        register != 1
        or sorted(declared) != list(range(len(declared)))
        or sorted(used) != sorted(declared)
    ):
        raise ValueError("QASM parameters/register do not match the supplied contract")
    return gates, len(declared)


def apply_qasm(weights, gates):
    operations = {"rx": qml.RX, "ry": qml.RY, "rz": qml.RZ}
    for name, index, wires in gates:
        if name == "cz":
            qml.CZ(wires=wires)
        else:
            operations[name](weights[index], wires=wires[0])


def block(theta, pair, kind):
    if kind == "heisenberg":
        convolution(theta, pair)
    elif kind == "light":
        qml.RY(theta[0], wires=pair[0])
        qml.RY(theta[1], wires=pair[1])
        qml.IsingXX(theta[2], wires=pair)
    elif kind == "su4":
        qml.Rot(*theta[:3], wires=pair[0])
        qml.Rot(*theta[3:6], wires=pair[1])
        qml.IsingXX(theta[6], wires=pair)
        qml.IsingYY(theta[7], wires=pair)
        qml.IsingZZ(theta[8], wires=pair)
        qml.Rot(*theta[9:12], wires=pair[0])
        qml.Rot(*theta[12:15], wires=pair[1])
    else:
        raise ValueError("Unknown convolution block")


class ArchitectureModel(nn.Module):
    def __init__(
        self,
        architecture,
        classes,
        preprocessor,
        *,
        head="linear",
        probability_floor=1e-12,
    ):
        super().__init__()
        if classes not in (3, 4) or head not in ("linear", "born"):
            raise ValueError("Expected three/four classes and a supported head")
        if head == "born" and (
            architecture.family != "qcnn"
            or architecture.readout != "probabilities"
            or architecture.output_qubits != 2
        ):
            raise ValueError("Pure multiclass readout requires two QCNN output qubits")
        self.architecture, self.classes, self.head_kind = architecture, classes, head
        self.preprocess, self.probability_floor = preprocessor, probability_floor
        name = architecture.name
        gates = None
        if architecture.family == "qnn":
            shape = {
                "qnn_ring": (2, 8, 3),
                "qnn_strong": (2, 8, 3),
                "qnn_basic": (2, 8),
                "qnn_two_design": (36,),
            }.get(name)
            if name in QASM_FILES:
                gates, count = parse_qasm(QASM_ROOT / QASM_FILES[name])
                shape = (count,)
            if shape is None:
                raise ValueError("Unknown QNN architecture")
            self.quantum_weights = nn.Parameter(
                0.2 * torch.randn(shape, dtype=torch.float64)
            )
            features = 16
        elif architecture.family == "qcnn":
            stages = 3 if architecture.output_qubits == 1 else 2
            parameters = {"heisenberg": 7, "light": 3, "su4": 15}[architecture.block]
            self.conv_weights = nn.Parameter(
                0.2 * torch.randn(stages, 2, parameters, dtype=torch.float64)
            )
            self.pool_weights = nn.Parameter(
                0.2 * torch.randn(stages, 3, dtype=torch.float64)
            )
            features = 4 if architecture.readout == "probabilities" else 3
        else:
            raise ValueError("Unknown model family")
        self.head = (
            nn.Linear(features, classes, dtype=torch.float64)
            if head == "linear"
            else None
        )
        device = qml.device("default.qubit", wires=8, shots=None)

        def circuit(angles, weights, pools=None):
            encode(angles, 8, "ry")
            if architecture.family == "qnn":
                if name in QASM_FILES:
                    apply_qasm(weights, gates)
                elif name == "qnn_basic":
                    qml.BasicEntanglerLayers(weights, wires=range(8), rotation=qml.RY)
                elif name == "qnn_strong":
                    qml.StronglyEntanglingLayers(
                        weights, wires=range(8), ranges=[1, 2], imprimitive=qml.CNOT
                    )
                elif name == "qnn_two_design":
                    qml.SimplifiedTwoDesign(
                        weights[:8], weights[8:].reshape(2, 7, 2), wires=range(8)
                    )
                else:
                    for layer in range(2):
                        for wire in range(8):
                            qml.Rot(*weights[layer, wire], wires=wire)
                        for wire in range(8):
                            qml.CNOT(wires=[wire, (wire + 1) % 8])
                return tuple(qml.expval(qml.PauliZ(w)) for w in range(8)) + tuple(
                    qml.expval(qml.PauliZ(w) @ qml.PauliZ((w + 1) % 8))
                    for w in range(8)
                )
            active = list(range(8))
            for stage in range(stages):
                for repeat in range(2):
                    for offset in (0, 1):
                        for k in range(offset, len(active) - 1, 2):
                            block(
                                weights[stage, repeat],
                                [active[k], active[k + 1]],
                                architecture.block,
                            )
                        if (
                            offset == 1
                            and architecture.topology == "periodic"
                            and len(active) > 2
                        ):
                            block(
                                weights[stage, repeat],
                                [active[-1], active[0]],
                                architecture.block,
                            )
                for k in range(0, len(active), 2):
                    pooling(pools[stage], active[k], active[k + 1])
                active = active[1::2]
            if architecture.readout == "probabilities":
                return qml.probs(wires=active)
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
            if self.architecture.family == "qnn"
            else self.qnode(angles, self.conv_weights, self.pool_weights)
        )
        return torch.stack(values, dim=-1) if isinstance(values, tuple) else values

    def forward(self, images):
        features = self.quantum_features(images)
        if self.head is not None:
            return self.head(features)
        probabilities = (
            features
            if self.classes == 4
            else torch.stack(
                (
                    features[..., 0],
                    features[..., 1],
                    features[..., 2] + features[..., 3],
                ),
                dim=-1,
            )
        )
        return probabilities.clamp_min(self.probability_floor).log()

    def predict_proba(self, images):
        return self(images).softmax(-1)

    def quantum_parameter_hash(self):
        import hashlib

        digest = hashlib.sha256()
        for name, value in self.named_parameters():
            if not name.startswith("head."):
                digest.update(name.encode())
                digest.update(value.detach().numpy().tobytes())
        return digest.hexdigest()

    def resources(self):
        args = (
            (torch.zeros((1, 8), dtype=torch.float64), self.quantum_weights)
            if self.architecture.family == "qnn"
            else (
                torch.zeros((1, 8), dtype=torch.float64),
                self.conv_weights,
                self.pool_weights,
            )
        )
        circuit = qml.workflow.construct_tape(self.qnode)(*args)
        # Count the unitary body without diagonalizing incompatible X/Y/Z
        # measurements. The simulator evaluates those analytically, separately.
        tapes, _ = qml.decompose(
            qml.tape.QuantumScript(circuit.operations, []),
            gate_set={
                "RX",
                "RY",
                "RZ",
                "Rot",
                "CZ",
                "CNOT",
                "CRY",
                "PauliX",
                "IsingXX",
                "IsingYY",
                "IsingZZ",
            },
        )
        tape = tapes[0]
        return {
            "quantum_parameters": sum(
                p.numel()
                for n, p in self.named_parameters()
                if not n.startswith("head.")
            ),
            "classical_parameters": sum(
                p.numel() for n, p in self.named_parameters() if n.startswith("head.")
            ),
            "gates": dict(Counter(op.name for op in tape.operations)),
            "two_wire_gates": sum(len(op.wires) == 2 for op in tape.operations),
            "output_qubits": self.architecture.output_qubits
            if self.architecture.family == "qcnn"
            else 8,
            "classical_layer_location": "after quantum measurement only"
            if self.head is not None
            else "none learned; fixed PCA and probability mapping remain",
        }
