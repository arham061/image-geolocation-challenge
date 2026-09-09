# Image Geolocation Challenge

An image geolocation project developed for my Deep Learning course. The model predicts latitude and longitude from a photograph using MobileViTV2, country classification, and geographic cells.

The final system achieved a **199.06 km median error on the validation split**, with approximately **4.44 million parameters**.

[Try the live demo](https://huggingface.co/spaces/arham061/Image-geolocation-challenge) · [Read the report](submission/report.pdf) · [Explore the notebooks](notebooks/)

## Results

| Metric | Validation result |
|---|---:|
| Median distance error | 199.06 km |
| Mean distance error | 497.73 km |
| Predictions within 200 km | 50.04% |
| Predictions within 750 km | 78.61% |

These results are from the validation split used during development. They are not scores on the unlabeled holdout set or a guarantee of accuracy on new uploads.

## How it works

The model predicts probabilities for 12 countries and 96 geographic cells, with eight cells per country.

Each cell's probability is multiplied by the probability of its country. After normalization, the model calculates coordinates as a weighted average of the cell centres.

The final prediction uses the same trained model at two input resolutions:

- 384 × 384 and 512 × 512 inputs
- Equal averaging of the two sets of output logits
- Country temperature of 0.25
- Cell temperature of 0.75

Temperature scaling changes how concentrated the probabilities are before country gating. Both resolutions use the same model weights.

## Experiments

The notebooks record the development process, including experiments that did not improve the results:

- Direct coordinate regression
- Auxiliary country classification
- Blending coordinates with country-centre estimates
- Geographic-cell classification
- Cell-only fine-tuning
- Country-gated cells
- Temperature scaling and gating ablations
- Balancing the number of training images per cell
- Higher-resolution training and fine-tuning
- Combining predictions at two resolutions

Increasing the input resolution produced some of the largest later improvements. Balanced cells and additional temperature tuning gave smaller or inconsistent gains.

The [report](submission/report.pdf) discusses the experiments and their limitations in more detail.

## Run the app locally

Clone the repository:

```bash
git clone https://github.com/arham061/image-geolocation-challenge.git
cd image-geolocation-challenge
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r submission/requirements.txt
python -m pip install gradio torchvision
```

Start the app:

```bash
python app.py
```

Open the local URL printed in the terminal. Upload a photograph to see the predicted country, coordinates, and an OpenStreetMap link.

The app uses CUDA when available and otherwise runs on CPU. It expects the exported model files in `submission/model/`.

The hosted Hugging Face version includes the additional ZeroGPU integration needed by that hosting environment.

## Submission files

The `submission/` directory contains:

- `model/`: exported weights, model configuration, preprocessing configuration, and inference settings
- `predict.py`: generates predictions for the holdout images
- `evaluate.py`: compares predictions with labeled coordinates
- `train.py`: training code
- `requirements.txt`: project dependencies
- `predictions.csv`: generated holdout predictions
- Report files

To run the holdout prediction script from the repository root:

```bash
python submission/predict.py
```

It expects images in `geo_dataset/holdout_public/` and writes `submission/predictions.csv`.

To evaluate predictions against a corresponding labels file:

```bash
python submission/evaluate.py \
    --labels path/to/labels.csv \
    --predictions path/to/predictions.csv
```

The labels CSV must contain `filename`, `lat`, and `lng`. The predictions CSV must contain `filename`, `pred_lat`, and `pred_lng`.

The dataset is not included in this repository.

## Limitations

The model was trained on images from Belarus, Finland, France, Germany, Iceland, Italy, Norway, Poland, Spain, Sweden, Turkey, and the United Kingdom.

It will still produce coordinates for images from other countries, but those predictions should not be considered reliable. Images with few geographic clues or different subjects and framing from the training data can also produce large errors.

The predicted country and final coordinates may disagree because the coordinates are averaged across multiple cell centres.

The validation split was used repeatedly to select experiments and inference settings, so the reported result may be optimistic compared with performance on unseen data.

## Training hardware

Training used an NVIDIA RTX 4000 Ada Generation GPU with approximately 20 GB of VRAM.

## Acknowledgements

The backbone is Apple's pretrained [MobileViTV2 model](https://huggingface.co/apple/mobilevitv2-1.0-imagenet1k-256), used through Hugging Face Transformers.

The demo uses Gradio and is hosted on Hugging Face Spaces. References and further methodological details are included in the report.

## License

See [LICENSE](LICENSE).
