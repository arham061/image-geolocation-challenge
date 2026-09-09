# European Image Geolocation

This project predicts the latitude and longitude of street-level photographs from 12 European countries. The final system fine-tunes a MobileViTV2 model to predict countries and balanced geographic cells, then converts the predicted cell probabilities into coordinates.

## Final method

The model produces 12 country logits and 96 geographic-cell logits. The cells are created separately inside each country using capacity-constrained K-means, with eight approximately equal-sized cells per country.

Country probabilities are used to reweight the probabilities of the cells belonging to each country. The final coordinate is the probability-weighted average of the 96 cell centres.

Training has two stages:

1. Train at 384 x 384 resolution with a learning rate of `1e-4`.
2. Load the best 384 model and fine-tune it at 512 x 512 with a learning rate of `3e-5`.

During final inference, the same saved model is evaluated at both 384 and 512 resolution. Their logits are averaged equally before applying country and cell temperature calibration, country gating, and coordinate calculation.

## Project structure

```text
Final Project/
├── geo_dataset/
│   ├── train/
│   ├── holdout_public/
│   └── train_labels.csv
└── submission/
    ├── model/
    │   ├── config.json
    │   ├── model.safetensors
    │   ├── preprocessor_config.json
    │   └── inference_config.pt
    ├── train.py
    ├── predict.py
    ├── evaluate.py
    ├── predictions.csv
    ├── requirements.txt
    └── README.md
```

The dataset is not included in the submission ZIP.

## Installation

Python 3.11 was used for the final experiments.

From the project root, create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r submission/requirements.txt
```

Use a CUDA-enabled PyTorch installation when running on an NVIDIA GPU.

## Training

Run the complete 384 and 512 training sequence from the project root:

```bash
python submission/train.py
```

The script performs the fixed stratified validation split, constructs the balanced cells, trains at 384 resolution, loads the best 384 checkpoint, and fine-tunes at 512 resolution. The best checkpoints and training histories are saved under `outputs/`. The final model is exported to `submission/model/`.

The submitted model has **4,444,245 parameters**.

The complete two-stage training process took approximately **two hours** on an **NVIDIA RTX 4000 Ada Generation GPU with 20 GB VRAM**. Training time may vary by hardware, and exact results can vary slightly because of stochastic optimization.

## Generate holdout predictions

The final trained model and inference configuration are already included. Generate `predictions.csv` with:

```bash
python submission/predict.py
```

The script reads all `.jpg` files from `geo_dataset/holdout_public/` and writes:

```text
submission/predictions.csv
```

The CSV contains exactly these columns:

```text
filename,pred_lat,pred_lng
```

## Evaluation

The evaluation script accepts a labeled CSV containing `filename`, `lat`, and `lng`, and a prediction CSV containing `filename`, `pred_lat`, and `pred_lng`:

```bash
python submission/evaluate.py \
    --labels path/to/labels.csv \
    --predictions path/to/predictions.csv
```

It reports mean and median Haversine distance, as well as the percentages of predictions within 200 km and 750 km.

The final local validation results were:

```text
Mean distance:   497.73 km
Median distance: 199.06 km
Within 200 km:    50.04%
Within 750 km:    78.61%
```

These results use equal 384/512 logit averaging, country temperature `0.25`, cell temperature `0.75`, and soft country gating.
