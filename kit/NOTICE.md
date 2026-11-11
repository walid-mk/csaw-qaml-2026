# Software, model and dataset notices

The organizer authorizes this public challenge release and its use by registered
participants in their private competition repositories. Public availability does
not select an additional open-source license for organizer code, trained
checkpoints or supplied circuit designs. Dataset and dependency licenses remain
separate. Keep your team's solutions and feedback private until both editions end.
The Road Signs images retain their separately stated CC0 terms.

The QNN implements the organizer-supplied/generated native eight-qubit RY/RZ
and CZ layout with three layers and a trained linear head. Its checkpoints were
trained locally; no external pretrained checkpoint is incorporated. The QCNN
uses a locally implemented SU4 convolution/pooling circuit and classical head.
Methodological references are Cong, Choi and Lukin, *Quantum convolutional neural
networks* (2019), and the PennyLane few-data tutorial by Kottmann, Mantilla
Calderon and Weber. The local circuit/readout/training details are defined by the
provided source; these references are not claims that their licenses cover all
organizer code or supplied QASM designs.

PennyLane and its official demonstrations are distributed under Apache License
2.0. Dependency software retains its own license; the challenge-use notice
above does not replace dependency or dataset terms. Dependencies are installed
from their original distributions rather than copied into this repository.

- [PennyLane license](https://github.com/PennyLaneAI/pennylane/blob/v0.45.0/LICENSE)
- [PennyLane demonstration repository](https://github.com/PennyLaneAI/demos)
- [Few-data tutorial](https://pennylane.ai/demos/tutorial_learning_few_data)
- [QCNN paper](https://doi.org/10.1038/s41567-019-0648-8)
- [Data attribution](DATA_ATTRIBUTION.md)
