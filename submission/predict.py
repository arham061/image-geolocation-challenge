from pathlib import Path

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)
from tqdm.auto import tqdm


MODEL_DIR = Path("submission/model")
INPUT_DIR = Path("geo_dataset/holdout_public")
OUTPUT_PATH = Path("submission/predictions.csv")

BATCH_SIZE = 8


class HoldoutDataset(Dataset):

    def __init__(
        self,
        image_directory,
        processor_384,
        processor_512,
    ):
        self.image_paths = sorted(
            image_directory.glob("*.jpg")
        )

        self.processor_384 = processor_384
        self.processor_512 = processor_512

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]

        image = Image.open(
            image_path
        ).convert("RGB")

        pixels_384 = self.processor_384(
            images=image,
            return_tensors="pt",
        )["pixel_values"].squeeze(0)

        pixels_512 = self.processor_512(
            images=image,
            return_tensors="pt",
        )["pixel_values"].squeeze(0)

        return (
            image_path.name,
            pixels_384,
            pixels_512,
        )


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


inference_config = torch.load(
    MODEL_DIR / "inference_config.pt",
    map_location="cpu",
    weights_only=True,
)

number_of_countries = (
    inference_config["number_of_countries"]
)

country_temperature = (
    inference_config["country_temperature"]
)

cell_temperature = (
    inference_config["cell_temperature"]
)

weight_384 = (
    inference_config["resolution_weights"][384]
)

weight_512 = (
    inference_config["resolution_weights"][512]
)


model = (
    AutoModelForImageClassification
    .from_pretrained(MODEL_DIR)
    .to(device)
)

model.eval()


processor_384 = AutoImageProcessor.from_pretrained(
    MODEL_DIR,
    size={
        "shortest_edge": 384,
    },
    crop_size={
        "height": 384,
        "width": 384,
    },
)

processor_512 = AutoImageProcessor.from_pretrained(
    MODEL_DIR,
    size={
        "shortest_edge": 512,
    },
    crop_size={
        "height": 512,
        "width": 512,
    },
)


dataset = HoldoutDataset(
    INPUT_DIR,
    processor_384,
    processor_512,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
)


cell_centres = (
    inference_config["cell_centres"]
    .to(device)
)

cell_to_country = (
    inference_config["cell_to_country"]
    .to(device)
)


prediction_rows = []

with torch.inference_mode():

    for (
        filenames,
        pixels_384,
        pixels_512,
    ) in tqdm(
        loader,
        desc="Generating predictions",
    ):

        pixels_384 = pixels_384.to(device)
        pixels_512 = pixels_512.to(device)

        # Run the same model at both resolutions
        logits_384 = model(
            pixel_values=pixels_384
        ).logits

        logits_512 = model(
            pixel_values=pixels_512
        ).logits

        # Combine both predictions
        combined_logits = (
            weight_384 * logits_384
            + weight_512 * logits_512
        )

        country_logits = combined_logits[
            :, :number_of_countries
        ]

        cell_logits = combined_logits[
            :, number_of_countries:
        ]

        # Apply the selected temperatures
        country_probabilities = torch.softmax(
            country_logits / country_temperature,
            dim=1,
        )

        cell_probabilities = torch.softmax(
            cell_logits / cell_temperature,
            dim=1,
        )

        # Use the country prediction to adjust
        # the cell probabilities
        country_weights = country_probabilities[
            :, cell_to_country
        ]

        gated_cell_probabilities = (
            cell_probabilities
            * country_weights
        )

        gated_cell_probabilities = (
            gated_cell_probabilities
            / gated_cell_probabilities.sum(
                dim=1,
                keepdim=True,
            )
        )

        # Calculate coordinates from cell centres
        normalized_coordinates = (
            gated_cell_probabilities
            @ cell_centres
        )

        coordinates = (
            normalized_coordinates
            * torch.tensor(
                [90.0, 180.0],
                device=device,
            )
        )

        coordinates = coordinates.cpu().numpy()

        for filename, coordinate in zip(
            filenames,
            coordinates,
        ):
            prediction_rows.append({
                "filename": filename,
                "pred_lat": coordinate[0],
                "pred_lng": coordinate[1],
            })


predictions = pd.DataFrame(
    prediction_rows
)

predictions.to_csv(
    OUTPUT_PATH,
    index=False,
)

print("Saved predictions to:", OUTPUT_PATH)
print("Number of predictions:", len(predictions))
print(predictions.head())