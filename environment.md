# Reference environment

The archived runs used Python 3.10.9 on Windows 10 with an NVIDIA RTX 3090.
Package versions are listed in `requirements.txt`.  CUDA wheels must be
installed from the PyTorch channel appropriate to the local CUDA setup;
the `+cu124` wheels are not generally available from the default PyPI
index.  No package installation is performed by this repository.

CPU-only verification of the archived tables does not require a GPU.
Re-running detector training or inference additionally requires the
original PhenoLeaf-TS media, a compatible COCO Mask R-CNN checkpoint,
and the model artefacts that are intentionally not redistributed here.
