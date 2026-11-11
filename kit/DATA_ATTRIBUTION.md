# Road Signs data attribution

Source: [Road Sign Detection](https://www.kaggle.com/datasets/andrewmvd/road-sign-detection),
published by **andrewmvd**, dataset ID 671172, version 1, updated 24 May 2020.
The publisher's metadata identifies **CC0: Public Domain** and credits
[Make ML, Road Signs Dataset](https://makeml.app/datasets/road-signs).
License: [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).
The license declaration was checked against public Kaggle metadata on 8 November 2026.

Organizer transformations: annotated object crops, conversion to RGB 32×32,
normalization to floating-point [0,1], source-group-based splits, fixed training-fit
preprocessing, and organizer-assigned incorrect target classes for attack inputs.
The existing pixels, class mapping, split assignments and fitted preprocessing
have not been changed for this kit.

The four classes are speed limit, crosswalk, physical traffic light, and stop.
Only the approved training records and exposed attack-input variants are included.
No entire source archive, reserved split, Teacher data, or BloodMNIST data is included.

The source page's separate promotional splash image is not distributed. CC0 is
the publisher's dataset declaration; it does not create an endorsement by the
dataset creators or a guarantee about rights unrelated to copyright.
