from argparse import ArgumentParser

import numpy as np
import pandas as pd


def haversine_km(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(
        np.radians,
        (lat1, lng1, lat2, lng2),
    )

    distance = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin((lng2 - lng1) / 2) ** 2
    )

    return (
        2
        * 6371.0088
        * np.arcsin(
            np.sqrt(
                np.clip(distance, 0, 1)
            )
        )
    )


parser = ArgumentParser()

parser.add_argument(
    "--labels",
    required=True,
)

parser.add_argument(
    "--predictions",
    required=True,
)

args = parser.parse_args()


labels = pd.read_csv(args.labels)
predictions = pd.read_csv(args.predictions)

data = labels.merge(
    predictions,
    on="filename",
)

distances = haversine_km(
    data["lat"],
    data["lng"],
    data["pred_lat"],
    data["pred_lng"],
)

print(f"Images: {len(distances)}")
print(f"Mean distance: {distances.mean():.2f} km")
print(f"Median distance: {np.median(distances):.2f} km")
print(f"Within 200 km: {(distances < 200).mean():.2%}")
print(f"Within 750 km: {(distances < 750).mean():.2%}")