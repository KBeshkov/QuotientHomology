# Quotient Homology of Neural Network Representations

Code accompanying the paper *A Quotient Homology Theory of Representation in Neural Networks*.

## Overview

ReLU networks are piecewise-linear maps that partition input space into convex polyhedra, on each of which the network acts as a fixed affine map. When the images of different polyhedra overlap in the output, the network effectively *glues* those regions together — collapsing topological structure.

This library makes that process precise and computable. It defines the **overlap decomposition** O_Φ — the partition of a dataset induced by which polyhedra have overlapping images — and proves that the homology of the network's representation is isomorphic to the quotient homology of the input with respect to O_Φ:

> H_k(Φ(M)) ≅ H_k(M / O_Φ)

This separates *topological* simplification (captured by the quotient) from *geometric* distortion (captured by the metric), giving a cleaner picture of how deep networks gradually destroy input topology than persistent homology methods provide.

## Installation

Requires [uv](https://github.com/astral-sh/uv) and the `cddlib` C library (`pycddlib` builds against it).

**macOS**
```bash
brew install cddlib
uv venv && uv pip install -e .
```

On Apple Silicon the Homebrew prefix is not in the default compiler search path, so pass the flags explicitly:
```bash
brew install cddlib
uv venv
CFLAGS="-I/opt/homebrew/include" LDFLAGS="-L/opt/homebrew/lib" uv pip install -e .
```

**Linux**
```bash
sudo apt install libcdd-dev   # Debian/Ubuntu
# or: sudo dnf install cddlib-devel
uv venv && uv pip install -e .
```

## Repository Structure

```
network-relative-homology/
├── src/
│   └── QuotientHomology/        # Installable package
│       ├── NetRelHom.py         # Neural network and decomposition classes
│       └── TopologicalMethods.py # Manifold generators and persistent homology
├── analysis/                    # Reproduction scripts for paper figures
│   ├── hom_propagation.py       # Quotient homology through layers (Naitzat datasets)
│   ├── top_propagation.py       # Train models on D-I / D-II / D-III
│   ├── num_overlap_test.py      # Overlap count vs. sphere dimension / width
│   ├── num_overlaps.py          # Volume and count statistics before/after training
│   ├── knot_test.py             # Relative homology on parametric knot mappings
│   ├── toroidal_test.py         # Polyhedral decomposition on torus inputs
│   └── full_region_analysis.py  # Full polyhedral region enumeration
├── data/                        # NumPy datasets (D-I, D-II, D-III)
├── models/                      # Pre-trained PyTorch model weights (30 × 5 configs)
└── pyproject.toml
```

## Core API

### `NetworkDecompositions(model)`

Analyses the polyhedral structure of a trained `FeedforwardNetwork`.

```python
from QuotientHomology import FeedforwardNetwork, NetworkDecompositions
import torch

model = FeedforwardNetwork(input_size=2, hidden_sizes=[25]*4, out_layer_sz=2)
X = torch.randn(500, 2)

decomp = NetworkDecompositions(model)
decomp.compute_overlap_decomp(X, sensitivity=10)

# decomp.overlap_decomposition[l]  — overlap classes at layer l
# decomp.polyhedral_decomposition[l] — polytope list at layer l
# decomp.global_decomposition       — codeword equivalence classes
```

### `PersistentHomology`

Computes standard and quotient/relative persistent homology.

```python
from QuotientHomology import PersistentHomology
from scipy.spatial.distance import cdist

Phom = PersistentHomology()

# Standard homology
dist, diagram, cycles = Phom.homology_analysis(X, cdist, dimred=False, params=[1, None])

# Quotient homology: pass the overlap decomposition as the submanifold
dist, diagram, cycles = Phom.relative_homology(X, decomp.overlap_decomposition[-1],
                                                cdist, dimred=False, params=[1, None])
```

### `ManifoldGenerator`

Samples point clouds from standard manifolds.

```python
from QuotientHomology import ManifoldGenerator

Mfld = ManifoldGenerator()
circle   = Mfld.S1(500, 1)          # 1-sphere
torus    = Mfld.T2(20, 1, 0.6)      # flat torus
knot     = Mfld.ParametricKnot(500, a=0.5, b=0.3, c=0.5, d=0.3, scale=[1])
spheres  = Mfld.Sn(500, [3])        # 2-sphere embedded in R^3
```

## Experiments

Scripts in `analysis/` reproduce the paper's figures. Each is self-contained and reads data from `../data/` relative to the script location.

| Script | Experiment |
|---|---|
| `top_propagation.py` | Train 30 models each on D-I (disks), D-II (linked rings), D-III (spherical shells) |
| `hom_propagation.py` | Compute layer-wise quotient Betti numbers and compare to Naitzat et al. |
| `num_overlap_test.py` | Overlap region count vs. sphere dimension, before and after training |
| `num_overlaps.py` | Volume of overlapping polyhedra before and after training |
| `knot_test.py` | Quotient homology of a parametric knot mapped through a 3-layer network |
| `toroidal_test.py` | Full polyhedral decomposition of a network trained on torus data |

Datasets D-I / D-II / D-III follow the protocol of [Naitzat et al. (2020)](https://jmlr.org/papers/v21/20-098.html): 9 disjoint instances of disks, linked ring pairs, and doubly-concentric spherical shells respectively, with Betti numbers β = (9,0,0), (9,9,0), and (9,0,9).

## Dependencies

| Package | Purpose |
|---|---|
| `torch` | Neural network models |
| `gudhi` | Simplicial complex construction |
| `ripser` / `persim` | Persistent homology and diagram comparison |
| `polytope` / `pycddlib` | H-representation of polyhedra, volume computation |
| `scipy` | Distance matrices, linear programming (overlap detection) |
| `scikit-learn` | Dimensionality reduction (Isomap, PCA) |
| `networkx` | Graph-based visualisation |

## Citation

```bibtex
@article{beshkov2026quotient,
  title   = {A Quotient Homology Theory of Representation in Neural Networks},
  author  = {Beshkov, Kosio},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
}
```
