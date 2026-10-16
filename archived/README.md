# Archived Materials

This directory preserves exploratory materials outside the current report's main scope. No experimental content was deleted. Each moved file was verified using SHA-256; `move_manifest.json` records its original path, archive path, and hash for lookup or restoration.

| Path | Contents |
| --- | --- |
| `configs/experiments.py`, `configs/mnist/` | Early configurations and MNIST experiments |
| `configs/cifar10/` | Older configurations for individual CIFAR-10 classical/PCA attack experiments |
| `models/mnist_cnn.py` | Legacy MNIST model |
| `notebooks/` | Single-PC and weighted PCA experiments, early attack comparisons, and CIFAR-10 analysis |
| `results/mnist/` | Formerly `results/archived/`; preserves early results and diagnostics with their internal directory structure |
| `results/cifar10/` | Former singular directories `classic_attack/`, `no_attack/`, and `pca_attack/` |
| `scripts/` | Obsolete `run.sh` and `run_cifar10_configs.sh` scripts |
| `placeholders/` | Previously empty environment and error files |
| `cache/` | Old Python caches and the tests directory, which contained only cached files |
| Existing `.py` files and `sign_comparison.png` at the archive root | Analysis scripts and a plot archived before this cleanup |

Archived files retain their contents from the time of the move. Their legacy imports, relative data paths, and output paths have not been rewritten, and compatibility with the current core code has not been verified. Existing notebook outputs remain available for inspection. Before rerunning an archived notebook, use the manifest to locate the moved data and update its references.

The current `main.py` registers only the three main experiment groups and additional seed configurations; it no longer loads the historical configurations stored here.
