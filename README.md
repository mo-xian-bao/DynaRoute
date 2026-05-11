<div align="center">
  <h2><b>(ICIC 2026 Accepted) DynaRoute: Dynamics-Conditioned Routing for Universal Time Series Forecasting</b></h2>
</div>

<div align="center">
## Updates / News

🚩 **News**: DynaRoute has been accepted by **ICIC 2026**.

## Introduction

DynaRoute is a dynamics-conditioned sparse forecasting model for universal time series forecasting. It keeps the efficient decoder-only sparse-expert forecasting pipeline, while introducing a compact global dynamics token inferred from the historical context.

The dynamics token is used to condition sparse expert routing and, optionally, the final prediction head. This design is intended to improve routing stability under nuisance transformations, long-horizon forecasting, and heterogeneous temporal regimes without changing the autoregressive decoding interface.

Core components:

- **Dynamics token inference**: single-query attention pooling over hidden trajectories.
- **Discrete-continuous dynamics representation**: straight-through vector-quantized prototype plus continuous residual.
- **Dynamics-conditioned routing**: sparse expert gates use both token states and the global dynamics token.
- **Dynamics-conditioned decoding**: optional feature-wise conditioning in the output head.
- **Dynamics-supervised objectives**: forecasting loss plus consistency, separation, and routing-stability losses.

## Training Data

DynaRoute supports common local time-series data formats through the dataset utilities in `dyna_route.datasets`.

Each training sequence can be stored in `jsonl`, `json`, `npy`, `npy.gz`, or `pkl` format. A simple `jsonl` dataset looks like this:

```json
{"sequence": [1.0, 2.0, 3.0, 4.0]}
{"sequence": [11.0, 22.0, 33.0, 44.0]}
```

Here is an example of loading a random sequence:

```python
import random
from dyna_route.datasets.dyna_route_dataset import DynaRouteDataset

ds = DynaRouteDataset("path/to/your_dataset")
seq_idx = random.randint(0, len(ds) - 1)
seq = ds[seq_idx]
```

First prepare your dataset locally, then instantiate `DynaRouteDataset` with the dataset file or folder path.

## Getting Started

### Installation

Install Python 3.10+, then install the dependencies:

```bash
pip install -r requirements.txt
```

The current implementation is tested with modern `transformers` releases and uses the dependency versions specified in `requirements.txt`.

On some Windows Python environments, duplicated OpenMP runtimes may raise an error. If that happens, set:

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

### Optional: Flash Attention

If your CUDA, PyTorch, and platform support Flash Attention, you may install it for faster training and inference:

```bash
pip install flash-attn==2.6.3
```

or:

```bash
pip install packaging
pip install ninja
MAX_JOBS=64 pip install flash-attn==2.6.3 --no-build-isolation
```

Use eager attention on CPU or unsupported platforms.

## Making Forecasts

Use a local DynaRoute checkpoint or a local DynaRoute config directory.

```python
import torch
from dyna_route.models.modeling_dyna_route import DynaRouteForPrediction

context_length = 12
seqs = torch.randn(2, context_length)  # [batch_size, context_length]

model = DynaRouteForPrediction.from_pretrained(
    "path/to/dynaroute_checkpoint",
    device_map="cpu",
)

# normalize seqs
mean = seqs.mean(dim=-1, keepdim=True)
std = seqs.std(dim=-1, keepdim=True).clamp_min(1e-6)
normed_seqs = (seqs - mean) / std

# forecast
prediction_length = 6
output = model.generate(normed_seqs, max_new_tokens=prediction_length)
normed_predictions = output[:, -prediction_length:]

# inverse normalize
predictions = normed_predictions * std + mean
```

If the sequences are already normalized:

```python
import torch
from dyna_route.models.modeling_dyna_route import DynaRouteForPrediction

context_length = 12
normed_seqs = torch.randn(2, context_length)

model = DynaRouteForPrediction.from_pretrained(
    "path/to/dynaroute_checkpoint",
    device_map="cpu",
)

prediction_length = 6
output = model.generate(normed_seqs, max_new_tokens=prediction_length)
normed_predictions = output[:, -prediction_length:]
```

For a quick local sanity check without a pretrained checkpoint, run:

```bash
python scripts/smoke_dyna_route.py
```

## Evaluation

Prepare benchmark datasets locally, for example under `./dataset`.

Example: evaluate on `ETTh1` with prediction horizon 96:

```bash
python run_eval.py -m path/to/dynaroute_checkpoint -d dataset/ETT-small/ETTh1.csv -p 96
```

If `--context_length` is not provided, the script uses the following defaults:

- `96 -> 512`
- `192 -> 1024`
- `336 -> 2048`
- `720 -> 3072`
- otherwise `prediction_length * 4`

## Fine-Tuning DynaRoute

### Preparing Your Dataset

Convert your time series into a compatible format. For `jsonl`, each line should be a dictionary with a `sequence` field:

```json
{"sequence": [1.0, 2.0, 3.0]}
{"sequence": [11.0, 22.0, 33.0]}
```

If your dataset is small, it is recommended to set `--stride 1` so the model sees more training windows.

### CPU Training

Replace `DATA_PATH` and `MODEL_PATH` with your local dataset and local DynaRoute config/checkpoint directory:

```bash
python main.py -d DATA_PATH -m MODEL_PATH --use_dyna_route --from_scratch
```

For a quick low-memory run, reduce the dynamics dimensions:

```bash
python main.py -d DATA_PATH -m MODEL_PATH --use_dyna_route --from_scratch --dyna_route_codebook_size 8 --dyna_route_code_dim 16 --dyna_route_residual_dim 16 --dyna_route_router_dim 16
```

### Single Node with Single or Multiple GPUs

```bash
python torch_dist_run.py main.py -d DATA_PATH -m MODEL_PATH --use_dyna_route
```

### Multi-Node Multi-GPU Training

```bash
export MASTER_ADDR=MASTER_ADDR
export MASTER_PORT=MASTER_PORT
export WORLD_SIZE=WORLD_SIZE
export RANK=RANK

python torch_dist_run.py main.py -d DATA_PATH -m MODEL_PATH --use_dyna_route
```

To train from scratch, add `--from_scratch`:

```bash
python torch_dist_run.py main.py -d DATA_PATH -m MODEL_PATH --use_dyna_route --from_scratch
```

For all available arguments:

```bash
python main.py --help
```

## Citation

Please cite DynaRoute if you find this repository helpful. Update the author and proceedings fields according to the camera-ready version:

```bibtex
@inproceedings{dynaroute2026,
  title     = {DynaRoute: Dynamics-Conditioned Routing for Universal Time Series Forecasting},
  author    = {To be updated},
  booktitle = {Proceedings of the International Conference on Intelligent Computing},
  year      = {2026},
  note      = {Accepted by ICIC 2026}
}
```
