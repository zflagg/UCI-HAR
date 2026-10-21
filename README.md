# Sensor Activity Recognition

Classifies human activity from raw multichannel wearable sensor data (accelerometer + gyroscope) with a 1D-CNN, compared against baselines and stress-tested against simulated failures.

> **Status:** scaffold — results below are placeholders until runs are complete.

## Results

Test set = subjects never seen in training.

| Model | Accuracy | Macro-F1 |
| --- | --- | --- |
| Majority class | 0.1822 | 0.8480 |
| Random forest (hand-crafted features) | 0.8510 | 0.8480 |
| 1D-CNN | | |
| 1D-CNN + augmentation | | |
| GRU | | |

### Robustness (macro-F1 under simulated sensor failures)

| Perturbation | Level | CNN | CNN + aug |
| --- | --- | --- | --- |
| Gaussian noise | σ = 0.5 | | |
| Channel dropout | p = 0.25 | | |
| Calibration drift | slope = 1.0 | | |

## What I tried and what I learned


- 
- 

## Design decisions

- **Subject-wise splits.** Validation and test sets contain only subjects absent from training. Windows overlap 50%, so a random window-level split would leak near-duplicate samples and overstate accuracy.
- **Train-only normalization.** Per-channel mean/std are fit on the training split and stored in the checkpoint, so inference uses the same statistics.
- **Macro-F1 as the main metric.** It weights every activity equally, so a model can't look good by only getting the common classes right.
- **Robustness testing.** Real sensors add noise, drop out, and drift out of calibration. `src/perturb.py` simulates all three, both as training augmentation and as test-time stress tests.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

[UCI HAR dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)

## Usage

Plumbing check on synthetic data:

```bash
python -m src.train --dataset synthetic --epochs 5
```

Full pipeline on UCI HAR:

```bash
DATA="data/UCI HAR Dataset"

python -m src.baseline   --dataset har --data-root "$DATA"
python -m src.train      --dataset har --data-root "$DATA" --model cnn
python -m src.train      --dataset har --data-root "$DATA" --model cnn --augment
python -m src.train      --dataset har --data-root "$DATA" --model gru
python -m src.robustness --dataset har --data-root "$DATA" --checkpoint runs/<run>/model.pt
python -m src.predict    --checkpoint runs/<run>/model.pt --csv recording.csv
```

Each training run writes `model.pt` and `results.json` (metrics, confusion matrix, per-epoch history) to its own folder under `runs/`.

## Project layout

```
src/
  data.py        loading, windowing, subject-wise splits, normalization
  models.py      1D-CNN and GRU classifiers
  engine.py      training loop, prediction, metrics
  perturb.py     noise / channel dropout / drift (augmentation + stress tests)
  train.py       training entry point with early stopping
  baseline.py    majority-class and random-forest baselines
  robustness.py  evaluate a checkpoint under sensor failures
  predict.py     run a checkpoint on a raw CSV recording
```

## Next steps

- 
