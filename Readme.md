# Weighted PCA Byzantine Attack: Code Guide

This project simulates Byzantine model poisoning in federated learning on CIFAR-10. It compares weighted PCA direction search with classical attacks under different aggregation rules, with and without Nearest Neighbor Mixing (NNM).

This README describes **the behavior of the current code**. See [Presentation.pdf](Presentation.pdf) for the presentation; and [archived/README.md](archived/README.md) for historical experiments.

## 1. Code Map

| File | Purpose and main entry points |
| --- | --- |
| [main.py](main.py) | CLI entry point; combines `CONFIGS`, builds the model, attack, and aggregators, runs training (50 rounds by default), and writes CSV output. Main functions: `run_federated_experiment`, `build_attack`, `build_aggregator`, and `build_log_row`. |
| [federated.py](federated.py) | `train_local_model` trains one benign client; `run_federated_round` handles attack generation, NNM, aggregation, and the global update; `evaluate_model` computes test loss and accuracy. |
| [pca_subspace_attack.py](pca_subspace_attack.py) | `PCASubspaceAttack`: SVD of benign updates, component weighting, sign search, Min-Max magnitude search, and candidate scoring. |
| [attacks.py](attacks.py) | Shared `Attack` interface and implementations of `ALIEOptim`, `MinMax`, `MinSum`, and `SignFlip`. Also retains `ALIE` and `OnesAttack`. |
| [aggregators.py](aggregators.py) | Aggregation functions and selection/contribution diagnostics. Main experiments use FedAvg, Median, Krum, Multi-Krum, and the variant currently named TrimmedMean. Bulyan and Prodigy implementations are also retained. |
| [preprocessing.py](preprocessing.py) | `nearest_neighbor_indices` and `apply_nnm_with_contributions`: mix nearest neighbors and track contributions from original clients. |
| [data.py](data.py) | CIFAR-10 loading and augmentation, IID/Dirichlet partitioning through `find_distribution_map`, and client DataLoaders. Also retains MNIST and shard partitioning utilities unused by the current entry point. |
| [sampling.py](sampling.py) | Stratified client sampling with a fixed malicious client count per round; checks participation requirements for Krum, Multi-Krum, and Bulyan. |
| [utils.py](utils.py) | Seed initialization, device selection, model/vector conversion, and CSV initialization/appending. |
| [models/cifar10_net.py](models/cifar10_net.py) | Defines `CIFAR10Simplified`, used by the current entry point, and `CIFAR10Net`, which is not instantiated by it. |
| [configs/base.py](configs/base.py) | Shared `Config` dataclass inherited and overridden by experiment configurations. |

Suggested reading order: `main.py` → `federated.py` → `pca_subspace_attack.py` → `preprocessing.py` / `aggregators.py` → configs → notebooks.

## 2. Training Flow

```text
main.py: select config, set seed, load CIFAR-10, partition client data
  └─ run_federated_experiment
       ├─ Create CIFAR10Simplified and the attack
       ├─ Initialize result CSV and record round 0
       └─ Each round: sample clients, build attacked/clean aggregators
            └─ run_federated_round
                 ├─ Train benign clients from the same global model
                 ├─ update = local parameters - global parameters
                 ├─ attack.generate: construct malicious updates from benign updates
                 ├─ Combine updates and optionally apply NNM
                 ├─ Aggregate and compute attack/selection diagnostics
                 └─ global parameters += aggregated update
            └─ Evaluate on the CIFAR-10 test set and append a CSV row
```

The code operates on **model updates**, rather than individual gradients. `train_local_model` uses SGD and CrossEntropyLoss. `utils.model_to_vector` flattens model parameters into CPU vectors, which are aggregated and written back to the model. Local training can use CUDA/MPS, while update vectors, PCA, and aggregation remain on the CPU.

The attacker receives all benign updates in the current round, and all malicious clients submit the same attack vector. This models a coordinated attacker with knowledge of the current benign updates and the target aggregation rule.

## 3. Environment and Data

Use the existing `adaptiveFL` Conda environment with Python 3.10 or later. Main dependencies:

| Purpose | Packages |
| --- | --- |
| Training and data | `torch`, `torchvision`, `numpy` |
| Analysis and plotting | `pandas`, `matplotlib` |
| Notebook execution | JupyterLab or a notebook-capable editor with the corresponding Python kernel |

`requirements.txt` currently lists only `matplotlib` and `pandas`; **it is not a complete environment specification**. The project does not yet provide a complete dependency lockfile.

```bash
conda activate adaptiveFL
python -c "import torch, torchvision, numpy, pandas, matplotlib; print('Dependencies import successfully')"
python main.py --help
```

The device is selected automatically in the order `CUDA → MPS → CPU`. There is no CLI device option.

By default, CIFAR-10 must already be extracted under:

```text
data/
└── cifar-10-batches-py/
    ├── data_batch_1 ... data_batch_5
    ├── test_batch
    └── batches.meta
```

`load_cifar10` uses `download=False`, so it does not download or extract the dataset. Training transforms include random cropping, horizontal flipping, and normalization; test transforms include tensor conversion and normalization only. `data/` is ignored by Git.

## 4. Experiment Configurations and Naming

`main.py` currently registers 211 configurations: 30 no-attack, 120 classical attack, 60 PCA attack, and one additional seed configuration.

| Configuration file | Experiment grid | Output directory |
| --- | --- | --- |
| [no_attack.py](configs/cifar10/no_attack.py) | 5 aggregators × raw/NNM × 3 seeds | `results/cifar10/no_attacks/` |
| [other_attacks.py](configs/cifar10/other_attacks.py) | 5 aggregators × 4 attacks × raw/NNM × 3 seeds | `results/cifar10/classic_attacks/` |
| [pca_attacks.py](configs/cifar10/pca_attacks.py) | 5 aggregators × K=2/3 × raw/NNM × 3 seeds | `results/cifar10/pca_attacks/` |
| [extra_seeds.py](configs/cifar10/extra_seeds.py) | Additional configuration `fedavg_min_max_raw_62` | `results/cifar10/extra_seeds/` |

Naming conventions:

```text
No attack:        {aggregator}_{raw|nnm}_{seed}
Classical attack: {aggregator}_{attack}_{raw|nnm}_{seed}
PCA:              {aggregator}_stdk{2|3}_p0_{raw|nnm}_{seed}
```

- Aggregator slugs: `fedavg`, `median`, `krum`, `multikrum`, `trimmedmean`.
- Classical attacks: `alie_optim`, `min_max`, `min_sum`, `sign_flip`.
- `stdk2` / `stdk3`: combine the first 2 / 3 PCs using standard deviation weights.
- `p0`: `pca_sign=0`, which enumerates component sign combinations; it does not mean zero perturbation.
- `raw`: aggregate directly; `nnm`: apply NNM before aggregation.
- Main seeds: `40`, `60`, `11`.

Effective settings for the main experiments:

| Parameter | Value and behavior |
| --- | --- |
| Clients | 10 total, 10 selected per round, malicious fraction 0.3 (3 malicious clients) |
| Rounds / local epochs | 50 / 1 |
| Batch size | 256 for local training, 512 for testing |
| Optimizer | SGD with momentum=0.9 |
| Learning rate | 0.05 for the first 33 rounds and 0.005 for the final 17, set inside `federated.py`; `Config.learning_rate=0.01` is currently overridden |
| Partition | Non-IID Dirichlet; `main.py` uses the `find_distribution_map` default α=0.5 |
| Model | Three Conv-ReLU-MaxPool blocks with 32/64/128 channels, followed by a 128-unit hidden layer and 10 output classes |
| Dropout | `CIFAR10Simplified` accepts `use_dropout` but currently contains no dropout layers |

Fields such as `shards_per_client` and the PCA noise options remain in the base config but are unused by the main experiment path. Change parameters in the corresponding dataclass or configuration generator. The CLI supports only `--config`; it does not provide overrides such as `--seed` or `--rounds`.

## 5. Running Experiments

Run all training commands from the **project root**. Each run resets its configured `output_csv`, overwriting any existing result at that path. Training does not resume from an interrupted round, and model checkpoints are not saved. To preserve existing results, give a new configuration a different `output_csv`.

### Single Experiment

```bash
# Default configuration: fedavg_raw_40
python main.py --config fedavg_raw_40

# Classical attack
python main.py --config median_min_sum_nnm_40

# PCA: K=2, dynamic sign search, without NNM
python main.py --config fedavg_stdk2_p0_raw_40

# PCA: K=3, dynamic sign search, with NNM
python main.py --config trimmedmean_stdk3_p0_nnm_11
```

### Batch Execution

```bash
# Run 120 classical attack and 60 PCA configurations sequentially
bash run_generated_attacks.sh

# List the 30 no-attack configurations without training
DRY_RUN=1 bash run_noattack.sh

# Run all 30 no-attack configurations
bash run_noattack.sh
```

Both scripts run configurations sequentially, record failed configuration names, and continue after failures. `run_generated_attacks.sh` uses the current `python` unless `PYTHON_BIN` is set. `run_noattack.sh` uses `PYTHON_BIN` if provided; otherwise, it tries `conda run -n adaptiveFL python` before falling back to `python`. Set `CONDA_ENV` to change its Conda environment name.

The default failure logs are `generated_attack_errors.txt` and `no_attack_errors.txt`; set `ERROR_FILE` to use another path. Both scripts clear their error file on startup; **the no-attack dry run also clears its configured error file**. The existing `generated_attack_errors.txt` records historical failed configuration names, without stack traces.

To stop a run at the first NaN/Inf detected by the finite-value checks:

```bash
STRICT_FINITE=1 python main.py --config fedavg_stdk2_p0_raw_40
```

With the default `STRICT_FINITE=0`, these checks print warnings and continue. They do not repair NaN/Inf values, and later operations may still fail.

### Adding a Configuration

For example, add a class to `configs/cifar10/extra_seeds.py` and register it in that file's `CONFIGS`. `main.py` already imports this registry:

```python
from configs.cifar10.pca_attacks import CONFIGS as PCA_CONFIGS

@dataclass
class FedAvgPCAK2Raw99(PCA_CONFIGS["fedavg_stdk2_p0_raw_40"]):
    seed: int = 99
    output_csv: str = "./results/cifar10/extra_seeds/fedavg_stdk2_p0_raw_99.csv"

CONFIGS["fedavg_stdk2_p0_raw_99"] = FedAvgPCAK2Raw99
```

The file already imports `dataclass`; preserve its existing `CONFIGS` entries. The new configuration can be run through `main.py --config`, but the two batch scripts and analysis notebooks do not automatically include the extra-seed registry or directory.

## 6. Attack, Aggregation, and NNM Implementations

### Weighted PCA

`PCASubspaceAttack.generate` receives `benign_updates` with shape `[number of benign clients, number of model parameters]`. PCA is recomputed every round:

1. Subtract the benign mean μ and apply SVD to the centered update matrix.
2. Take the first K principal components vᵢ, with λᵢ = singular_valueᵢ² / (number of benign clients − 1).
3. In `std_weighted` mode, set wᵢ = √λᵢ and construct a unit direction `d = normalize(Σ sᵢwᵢvᵢ)` for each sign combination.
4. Construct `m = μ − γd` and search for the largest γ satisfying `maxᵢ ‖m−bᵢ‖₂ ≤ maxᵢⱼ ‖bᵢ−bⱼ‖₂`.
5. Score candidates using `‖A(B ∪ M) − A(B)‖₂`; select the largest score and return repeated malicious vectors with diagnostics.

K=2 and K=3 yield 4 and 8 sign candidates, respectively. `sign=+1/-1` fixes the sign of every active component; main configurations use `sign=0`. Direction selection does not access test accuracy.

PCA candidate scoring currently **does not include NNM**. It also uses the same aggregator, with the same bound malicious count f, for both the benign-only and combined collections. This differs from the `attack_aggregation_shift` diagnostic computed separately in the training flow.

### Classical Attacks

| Class | Current behavior |
| --- | --- |
| `MinMax` | Searches along `−sign(μ)` under a maximum-distance constraint between the malicious vector and benign vectors; γ is searched over [0,10]. |
| `MinSum` | Uses the same direction with a sum-distance constraint. The implementation sums Euclidean distances and searches γ over [0,10]. |
| `ALIEOptim` | Searches candidates `m = μ − zσ`, using population standard deviation σ. Defaults: z from 0.125 to 2.0 in steps of 0.125. Maximizes the distance from the aggregate to μ; candidate scoring includes NNM when enabled. |
| `SignFlip` | Sets `m = −scale × μ`, with scale=1 by default. |

### Aggregation and NNM

- `fed_avg` averages client updates equally; it does not weight them by client dataset size.
- `coordinate_wise_median` applies `torch.median` to each coordinate.
- Krum scores each candidate by its sum of squared distances to the nearest n−f−2 other vectors, then selects the lowest-scoring candidate.
- Multi-Krum uses the same scores, selects the lowest-scoring n−f−2 vectors, and averages them.
- `trimmed_mean` currently averages the **n−2f values closest to the median** in each coordinate. This differs from standard trimmed mean, which removes the f largest and f smallest values.
- NNM replaces each vector with the average of its n−f nearest neighbors, including itself, and returns a contribution matrix for diagnostics. It requires f < n/2.
- Bulyan and Prodigy remain available through the aggregator factory but are unused by the current 211 configurations. The main setting n=10, f=3 also fails Bulyan's participation requirement.

**No-attack baseline behavior:** three clients remain marked as malicious and are not trained as benign clients, so only seven benign updates are submitted. With no attack, the aggregator uses f=0. NNM therefore averages all seven updates for every output row, producing identical mixed vectors. This baseline does not represent ten benign clients.

## 7. CSV Output

Results include the initial evaluation at round 0 and an evaluation after every training round. A complete 50-round run has 51 rows. Raw accuracy values range from 0 to 1; notebook tables convert them to percentages.

| Field group | Meaning |
| --- | --- |
| `round`, `test_loss`, `test_accuracy`, `learning_rate` | Training progress, test metrics, and the effective learning rate for that round. |
| `top_k_eigs`, `gamma`, `selected_signs`, `sign_scores` | PCA eigenvalues, perturbation magnitude, selected signs, and candidate scores. List/dict fields are serialized as JSON. |
| `benign_update_norm_mean`, `malicious_update_norm_mean`, `global_update_norm` | Mean benign/malicious update norms and the final aggregated update norm. |
| `malicious_to_benign_mean_distance` / `cosine` | Distance and cosine similarity between the mean malicious and mean benign updates, measured before NNM. |
| `attack_aggregation_shift`, `relative_attack_shift`, `attack_clean_aggregation_cosine` | Differences between the attacked aggregate and a clean aggregate of the current round's benign updates. The clean aggregator uses f=0; with NNM enabled, the clean collection is preprocessed separately. This is not a separate no-attack training trajectory. |
| `aggregated_malicious_clients` | Number of original malicious clients assigned nonzero contribution, accounting for aggregation selection and NNM contributions; not a count of labels attached to mixed vectors. |
| `selected_benign_contribution`, `selected_malicious_contribution`, `selected_update_to_benign_mean_cosine` | Source contributions and direction diagnostics for vectors selected by Krum/Multi-Krum. |

Diagnostics may be empty at round 0, in no-attack runs, or when inapplicable. Older CSVs may lack fields added later.

## 8. Analyzing Existing Results

| Notebook | Purpose |
| --- | --- |
| [cifar10_final.ipynb](notebooks/cifar10_final.ipynb) | Loads the three main result groups and produces accuracy and distance/cosine/shift summary tables. |
| [cifar10_diagnostics_analysis.ipynb](notebooks/cifar10_diagnostics_analysis.ipynb) | Diagnostic tables and curves. `run_analysis(last_n=5, clip_quantile=0.95, show=True)` returns data, summary tables, and saved plot paths. |

Use a notebook kernel for the `adaptiveFL` environment and set the **working directory to `notebooks/`**, as required by the relative paths. Execute cells in order. After defining the functions in the diagnostics notebook, run:

```python
analysis = run_analysis()
analysis["tables"][("test_accuracy_filled", "nnm")]
```

By default, analysis reads only seeds 40/60/11 from the three main directories. It averages rounds 46–50 within each seed, then computes the mean and sample standard deviation across seeds. Diagnostic plots are saved under `results/cifar10/plots/diagnostics/`. For metrics other than accuracy and cosine similarity, `clip_quantile` controls the displayed range and may place spikes outside the plot bounds.

The notebooks fill accuracy values for missing rounds with **0.1**. A reported 10% may therefore be measured accuracy or an imputed value after an interrupted run; it does not necessarily indicate stable convergence. The diagnostics notebook excludes NNM + Sign Flip by default, while the final notebook includes it, so their comparison scopes differ.

The current main results contain 210 CSVs: 176 have all 51 rows, while 34 runs ended early. Reports based on these results should disclose missing-value handling. See [RESEARCH_RECAP.md](RESEARCH_RECAP.md) for research limitations and discrepancies between the presentation and implementation.

## 9. Archive

The active workspace retains core code, main configurations, two analysis notebooks, main results, plots, and additional seed experiments. `archived/` contains MNIST experiments, early CIFAR-10 configurations/results, old notebooks, obsolete scripts, and caches. Imports and relative paths in archived files have not been fully updated; these files are historical references rather than current executable entry points.

See [archived/move_manifest.json](archived/move_manifest.json) for original paths, archive paths, and file hashes.
