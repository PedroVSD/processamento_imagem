"""PathoShift-CRC: reproducible pipeline for histopathology patch classification.

The official dataset (PathoShift-CRC) is a 11,250-image subset of
NCT-CRC-HE-100K-NONORM with nine tissue classes. Images are 224x224 RGB TIFFs
described by the ``train.csv`` / ``validation.csv`` manifests, and robustness
evaluation applies deterministic perturbations described by ``robustness.csv``.
"""

__version__ = "0.1.0"

#: Nine classes of the official dataset, in the order used for reports.
LABELS: tuple[str, ...] = (
    "ADI",
    "BACK",
    "DEB",
    "LYM",
    "MUC",
    "MUS",
    "NORM",
    "STR",
    "TUM",
)
