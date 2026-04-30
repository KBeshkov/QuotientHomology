# Quotient Homology of Neural Network Representations

## Overview
Repository for the paper **A Quotient Homology Theory of Representation Neural Networks**. Using the fact that ReLU networks split the input space into convex regions and apply a piecewise linear map on each such region, the paper proves that topologically impactful transformations can be described by two sources. The first (**Rank source**) is related to the rank of each such map, whereas the second (**Overlap source**) is related to how the images of multiple regions intersect. When the intersections between the data manifold and the convex regions are convex, we prove that the homology of the network's representation is isomorphic to the quotient homology of the input with respect to $O_\Phi$ or,

$H_k(\Phi(M)) ≅ H_k(M / O_\Phi)$.

This separates *topological* simplification (captured by the quotient) from *geometric* distortion (captured by the metric), giving a clean picture of how deep networks gradually change input topology.

## Installation

Requires [uv](https://github.com/astral-sh/uv) and the `cddlib` C library (`pycddlib` builds against it).

**macOS**
```bash
brew install cddlib
uv venv
uv pip install -e .
```

Sometimes the Homebrew prefix is not in the default compiler search path, so you might have to pass the flags explicitly:
```bash
brew install cddlib
uv venv
CFLAGS="-I/opt/homebrew/include" LDFLAGS="-L/opt/homebrew/lib" uv pip install -e .
```

**Linux**
```bash
sudo apt install libcdd-dev 
uv venv
uv pip install -e .
```

## Repository Structure

```
network-relative-homology/
├── src/
│   └── QuotientHomology/        
│       ├── NetRelHom.py          # Neural network and decomposition classes
│       └── TopologicalMethods.py # Manifold generators and persistent homology
├── analysis/                    
│   ├── hom_propagation.py        # Quotient homology of the Naitzat et al. data
│   ├── top_propagation.py        # Trains the models on the Naitzat et al. data
│   ├── num_overlap_test.py       # Overlap count and size for spheres of different dimension
│   ├── knot_test.py              # Quotient homology on nonlinear curves
│   ├── XOR_examples              # Plots of XOR examples
├── data/                         # Subsampled reproduction of the Naitzat et al. data
├── models/                       # Trained PyTorch model weights for all Naitzat et al. data
└── pyproject.toml
```


## Example usage

Analyses the polyhedral structure of a trained `FeedforwardNetwork`.

```python
from QuotientHomology import FeedforwardNetwork, NetworkDecompositions
import torch

#Genrate a network with an architecture of (2,10,10,2).
model = FeedforwardNetwork(input_size=2, hidden_sizes=[10]*2, out_layer_sz=2)
X = torch.randn(500, 2)

#Initialize and compute the overlap decomposition
decomp = NetworkDecompositions(model)
decomp.compute_overlap_decomp(X)

#Once computed you can access each decomposition at each layer:
# decomp.overlap_decomposition[l]  — overlap classes at layer l
# decomp.polyhedral_decomposition[l] — polyherdon list at layer l
# decomp.global_decomposition       — codeword equivalence classes
```

## Citation

```bibtex
@article{beshkov2026quotient,
  title   = {A Quotient Homology Theory of Representation in Neural Networks},
  author  = {Beshkov, Kosio},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
}
```
