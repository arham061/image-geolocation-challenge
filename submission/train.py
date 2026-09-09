from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from k_means_constrained import KMeansConstrained
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "geo_dataset"
TRAIN_DIR = DATA_DIR / "train"
LABELS_PATH = DATA_DIR / "train_labels.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FINAL_MODEL_DIR = Path(__file__).resolve().parent / "model"

MODEL_NAME = "apple/mobilevitv2-1.0-imagenet1k-256"
CELLS_PER_COUNTRY = 8
COUNTRY_LOSS_WEIGHT = 0.01
CELL_LOSS_WEIGHT = 0.01
PATIENCE = 4


def haversine_km(lat1, lng1, lat2, lng2):
    radius = 6371.0088
    lat1, lng1, lat2, lng2 = map(
        np.radians,
        (lat1, lng1, lat2, lng2),
    )
    difference = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin((lng2 - lng1) / 2) ** 2
    )
    return 2 * radius * np.arcsin(
        np.sqrt(np.clip(difference, 0, 1))
    )


def create_geographic_cells(train_df, val_df, country_to_index):
    number_of_cells = len(country_to_index) * CELLS_PER_COUNTRY
    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df["cell_index"] = -1
    val_df["cell_index"] = -1

    cell_centres = np.zeros((number_of_cells, 2), dtype=np.float32)
    cell_to_country = np.zeros(number_of_cells, dtype=np.int64)

    for country, country_index in country_to_index.items():
        train_mask = train_df["country"] == country
        val_mask = val_df["country"] == country

        train_coordinates = train_df.loc[
            train_mask, ["lat", "lng"]
        ].to_numpy(dtype=np.float32)
        val_coordinates = val_df.loc[
            val_mask, ["lat", "lng"]
        ].to_numpy(dtype=np.float32)

        longitude_scale = np.cos(
            np.radians(train_coordinates[:, 0].mean())
        )
        projected_train = train_coordinates.copy()
        projected_val = val_coordinates.copy()
        projected_train[:, 1] *= longitude_scale
        projected_val[:, 1] *= longitude_scale

        number_of_images = len(train_coordinates)
        minimum_size = number_of_images // CELLS_PER_COUNTRY
        maximum_size = int(np.ceil(number_of_images / CELLS_PER_COUNTRY))

        clustering = KMeansConstrained(
            n_clusters=CELLS_PER_COUNTRY,
            size_min=minimum_size,
            size_max=maximum_size,
            random_state=42,
            n_init=10,
            max_iter=100,
        )
        local_train_cells = clustering.fit_predict(projected_train)

        validation_distances = (
            (
                projected_val[:, None, :]
                - clustering.cluster_centers_[None, :, :]
            )
            ** 2
        ).sum(axis=2)
        local_val_cells = validation_distances.argmin(axis=1)

        first_cell = country_index * CELLS_PER_COUNTRY
        last_cell = first_cell + CELLS_PER_COUNTRY
        train_df.loc[train_mask, "cell_index"] = (
            first_cell + local_train_cells
        )
        val_df.loc[val_mask, "cell_index"] = first_cell + local_val_cells

        for local_cell in range(CELLS_PER_COUNTRY):
            global_cell = first_cell + local_cell
            assigned = train_coordinates[local_train_cells == local_cell]
            cell_centres[global_cell] = assigned.mean(axis=0)

        cell_to_country[first_cell:last_cell] = country_index

    train_df["cell_index"] = train_df["cell_index"].astype(int)
    val_df["cell_index"] = val_df["cell_index"].astype(int)
    return train_df, val_df, cell_centres, cell_to_country


class GeolocationDataset(Dataset):
    def __init__(self, dataframe, processor):
        self.dataframe = dataframe.reset_index(drop=True)
        self.processor = processor

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        with Image.open(TRAIN_DIR / row["filename"]) as image:
            image = image.convert("RGB")
            pixel_values = self.processor(
                images=image,
                return_tensors="pt",
            )["pixel_values"].squeeze(0)

        coordinates = torch.tensor(
            [row["lat"] / 90, row["lng"] / 180],
            dtype=torch.float32,
        )
        return (
            pixel_values,
            coordinates,
            torch.tensor(row["country_index"], dtype=torch.long),
            torch.tensor(row["cell_index"], dtype=torch.long),
        )


def make_loaders(train_df, val_df, processor, batch_size):
    train_loader = DataLoader(
        GeolocationDataset(train_df, processor),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        GeolocationDataset(val_df, processor),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader


def predict_coordinates(outputs, number_of_countries, cell_to_country, centres):
    country_logits = outputs[:, :number_of_countries]
    cell_logits = outputs[:, number_of_countries:]
    country_probabilities = torch.softmax(country_logits, dim=1)
    cell_probabilities = torch.softmax(cell_logits, dim=1)
    gated_probabilities = (
        cell_probabilities
        * country_probabilities[:, cell_to_country]
    )
    gated_probabilities = gated_probabilities / gated_probabilities.sum(
        dim=1,
        keepdim=True,
    ).clamp_min(1e-8)
    coordinates = gated_probabilities @ centres
    return coordinates, country_logits, cell_logits


def evaluate(
    model,
    loader,
    device,
    number_of_countries,
    cell_to_country,
    centres,
    coordinate_loss_function,
    country_loss_function,
    cell_loss_function,
):
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    correct_countries = 0
    correct_cells = 0
    number_of_images = 0

    with torch.inference_mode():
        for images, coordinates, country_labels, cell_labels in loader:
            images = images.to(device)
            coordinates = coordinates.to(device)
            country_labels = country_labels.to(device)
            cell_labels = cell_labels.to(device)

            outputs = model(pixel_values=images).logits
            final_coordinates, country_logits, cell_logits = predict_coordinates(
                outputs,
                number_of_countries,
                cell_to_country,
                centres,
            )
            loss = (
                coordinate_loss_function(final_coordinates, coordinates)
                + COUNTRY_LOSS_WEIGHT
                * country_loss_function(country_logits, country_labels)
                + CELL_LOSS_WEIGHT
                * cell_loss_function(cell_logits, cell_labels)
            )
            total_loss += loss.item()
            predictions.append(final_coordinates.cpu().numpy())
            targets.append(coordinates.cpu().numpy())
            correct_countries += (
                country_logits.argmax(dim=1) == country_labels
            ).sum().item()
            correct_cells += (
                cell_logits.argmax(dim=1) == cell_labels
            ).sum().item()
            number_of_images += len(images)

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    predictions *= np.array([90, 180])
    targets *= np.array([90, 180])
    distances = haversine_km(
        targets[:, 0],
        targets[:, 1],
        predictions[:, 0],
        predictions[:, 1],
    )
    return {
        "validation_loss": total_loss / len(loader),
        "mean_km": float(np.mean(distances)),
        "median_km": float(np.median(distances)),
        "within_200": float(np.mean(distances < 200)),
        "within_750": float(np.mean(distances < 750)),
        "country_accuracy": correct_countries / number_of_images,
        "cell_accuracy": correct_cells / number_of_images,
    }


def train_stage(
    model,
    train_df,
    val_df,
    device,
    number_of_countries,
    cell_to_country,
    centres,
    resolution,
    batch_size,
    learning_rate,
    maximum_epochs,
    output_directory,
    save_initial_model=False,
):
    output_directory.mkdir(parents=True, exist_ok=True)
    best_model_path = output_directory / "best_model.pt"
    history_path = output_directory / "history.csv"

    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME,
        size={"shortest_edge": resolution},
        crop_size={"height": resolution, "width": resolution},
    )
    train_loader, val_loader = make_loaders(
        train_df,
        val_df,
        processor,
        batch_size,
    )

    coordinate_loss_function = nn.MSELoss()
    country_loss_function = nn.CrossEntropyLoss()
    cell_loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_median = float("inf")
    if save_initial_model:
        initial = evaluate(
            model,
            val_loader,
            device,
            number_of_countries,
            cell_to_country,
            centres,
            coordinate_loss_function,
            country_loss_function,
            cell_loss_function,
        )
        best_median = initial["median_km"]
        torch.save(model.state_dict(), best_model_path)
        print(
            f"Inherited model at {resolution}: "
            f"{best_median:.1f} km median"
        )

    history = []
    epochs_without_improvement = 0

    for epoch in range(maximum_epochs):
        model.train()
        total_training_loss = 0
        training_bar = tqdm(
            train_loader,
            desc=f"{resolution}px epoch {epoch + 1}/{maximum_epochs}",
        )

        for images, coordinates, country_labels, cell_labels in training_bar:
            images = images.to(device)
            coordinates = coordinates.to(device)
            country_labels = country_labels.to(device)
            cell_labels = cell_labels.to(device)
            optimizer.zero_grad()

            outputs = model(pixel_values=images).logits
            final_coordinates, country_logits, cell_logits = predict_coordinates(
                outputs,
                number_of_countries,
                cell_to_country,
                centres,
            )
            loss = (
                coordinate_loss_function(final_coordinates, coordinates)
                + COUNTRY_LOSS_WEIGHT
                * country_loss_function(country_logits, country_labels)
                + CELL_LOSS_WEIGHT
                * cell_loss_function(cell_logits, cell_labels)
            )
            loss.backward()
            optimizer.step()
            total_training_loss += loss.item()
            training_bar.set_postfix(loss=f"{loss.item():.4f}")

        metrics = evaluate(
            model,
            val_loader,
            device,
            number_of_countries,
            cell_to_country,
            centres,
            coordinate_loss_function,
            country_loss_function,
            cell_loss_function,
        )
        metrics["epoch"] = epoch + 1
        metrics["training_loss"] = total_training_loss / len(train_loader)
        history.append(metrics)
        pd.DataFrame(history).to_csv(history_path, index=False)

        print(
            f"Median: {metrics['median_km']:.1f} km | "
            f"Mean: {metrics['mean_km']:.1f} km | "
            f"<200 km: {metrics['within_200']:.2%} | "
            f"<750 km: {metrics['within_750']:.2%} | "
            f"Country: {metrics['country_accuracy']:.2%} | "
            f"Cell: {metrics['cell_accuracy']:.2%}"
        )

        if metrics["median_km"] < best_median:
            best_median = metrics["median_km"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
            print("Saved new best model.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping.")
            break

    model.load_state_dict(
        torch.load(best_model_path, map_location=device, weights_only=True)
    )
    return processor, best_model_path


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    start_time = perf_counter()

    labels = pd.read_csv(LABELS_PATH)
    countries = sorted(labels["country"].unique())
    country_to_index = {
        country: index for index, country in enumerate(countries)
    }
    labels["country_index"] = labels["country"].map(country_to_index)
    train_df, val_df = train_test_split(
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels["country"],
    )
    train_df, val_df, cell_centres, cell_to_country = (
        create_geographic_cells(train_df, val_df, country_to_index)
    )

    number_of_countries = len(countries)
    number_of_cells = number_of_countries * CELLS_PER_COUNTRY
    normalized_centres = cell_centres.copy()
    normalized_centres[:, 0] /= 90
    normalized_centres[:, 1] /= 180

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    centres = torch.tensor(normalized_centres, dtype=torch.float32, device=device)
    cell_to_country_tensor = torch.tensor(
        cell_to_country,
        dtype=torch.long,
        device=device,
    )

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=number_of_countries + number_of_cells,
        ignore_mismatched_sizes=True,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert parameter_count <= 5_000_000
    print(f"Device: {device}")
    print(f"Parameters: {parameter_count:,}")

    train_stage(
        model=model,
        train_df=train_df,
        val_df=val_df,
        device=device,
        number_of_countries=number_of_countries,
        cell_to_country=cell_to_country_tensor,
        centres=centres,
        resolution=384,
        batch_size=16,
        learning_rate=1e-4,
        maximum_epochs=40,
        output_directory=OUTPUT_DIR / "high_resolution_384",
    )

    processor_512, _ = train_stage(
        model=model,
        train_df=train_df,
        val_df=val_df,
        device=device,
        number_of_countries=number_of_countries,
        cell_to_country=cell_to_country_tensor,
        centres=centres,
        resolution=512,
        batch_size=8,
        learning_rate=3e-5,
        maximum_epochs=15,
        output_directory=OUTPUT_DIR / "high_resolution_512_finetune",
        save_initial_model=True,
    )

    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(FINAL_MODEL_DIR, safe_serialization=True)
    processor_512.save_pretrained(FINAL_MODEL_DIR)
    torch.save(
        {
            "resolutions": [384, 512],
            "resolution_weights": {384: 0.50, 512: 0.50},
            "country_temperature": 0.25,
            "cell_temperature": 0.75,
            "number_of_countries": number_of_countries,
            "number_of_cells": number_of_cells,
            "cell_centres": centres.detach().cpu(),
            "cell_to_country": cell_to_country_tensor.detach().cpu(),
            "country_to_index": country_to_index,
            "parameter_count": parameter_count,
        },
        FINAL_MODEL_DIR / "inference_config.pt",
    )

    elapsed_hours = (perf_counter() - start_time) / 3600
    print(f"Training completed in {elapsed_hours:.2f} hours.")
    print(f"Final model saved to {FINAL_MODEL_DIR}")


if __name__ == "__main__":
    main()
