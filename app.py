import spaces
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification


MODEL_DIR = Path(__file__).resolve().parent / "submission" / "model"
device = torch.device("cuda")

config = torch.load(
    MODEL_DIR / "inference_config.pt",
    map_location="cpu",
    weights_only=True,
)

model = AutoModelForImageClassification.from_pretrained(
    MODEL_DIR, local_files_only=True,
).to(device)
model.eval()

processors = {
    resolution: AutoImageProcessor.from_pretrained(
        MODEL_DIR,
        size={"shortest_edge": resolution},
        crop_size={"height": resolution, "width": resolution},
        local_files_only=True,
    )
    for resolution in (384, 512)
}

number_of_countries = config["number_of_countries"]
cell_centres = config["cell_centres"].to(device)
cell_to_country = config["cell_to_country"].to(device)
index_to_country = {
    index: country.replace("_", " ")
    for country, index in config["country_to_index"].items()
}


@spaces.GPU
@torch.inference_mode()
def predict_image(image):
    if image is None:
        return "Please upload an image first."

    image = image.convert("RGB")

    # Use the same model at both resolutions.
    pixels_384 = processors[384](
        images=image, return_tensors="pt",
    )["pixel_values"].to(device)
    pixels_512 = processors[512](
        images=image, return_tensors="pt",
    )["pixel_values"].to(device)

    logits_384 = model(pixel_values=pixels_384).logits
    logits_512 = model(pixel_values=pixels_512).logits
    combined_logits = (
        config["resolution_weights"][384] * logits_384
        + config["resolution_weights"][512] * logits_512
    )

    country_probabilities = torch.softmax(
        combined_logits[:, :number_of_countries]
        / config["country_temperature"],
        dim=1,
    )
    cell_probabilities = torch.softmax(
        combined_logits[:, number_of_countries:]
        / config["cell_temperature"],
        dim=1,
    )

    # Reweight cells using their country's probability.
    gated_probabilities = (
        cell_probabilities * country_probabilities[:, cell_to_country]
    )
    gated_probabilities = gated_probabilities / gated_probabilities.sum(
        dim=1, keepdim=True,
    )

    coordinates = (gated_probabilities @ cell_centres) * torch.tensor(
        [90.0, 180.0], device=device,
    )
    latitude, longitude = coordinates[0].cpu().tolist()
    country_index = country_probabilities.argmax(dim=1).item()
    country = index_to_country[country_index]

    return (
        f"**Predicted country:** {country}\n\n"
        f"**Latitude:** {latitude:.5f}\n\n"
        f"**Longitude:** {longitude:.5f}\n\n"
        f"[View predicted location on OpenStreetMap](https://www.openstreetmap.org/"
        f"?mlat={latitude:.6f}&mlon={longitude:.6f}"
        f"#map=6/{latitude:.6f}/{longitude:.6f})\n\n"
        "The country is the country head's prediction; the averaged coordinate "
        "can fall outside that country's borders."
    )


demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", sources=["upload"], label="Upload a landscape or street image"),
    outputs=gr.Markdown(),
    title="Where was this photo taken?",
    description=(
        "A student image-geolocation project using MobileViTV2. "
        "Upload a photo to estimate its location from visual content. "
        "Validation median error: approximately 199 km, not a guarantee for new photos."
    ),
    article=(
        "Trained on images from: "
        + ", ".join(index_to_country[index] for index in sorted(index_to_country))
        + ". Images from other countries can still produce a prediction, "
        "but it should not be treated as reliable."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()

