import numpy as np
import pandas as pd
from helpers import (
    interpolate,
    flatten,
    bbox,
    center,
    sort_coords,
    four_corners,
    eight_corners,
    sixteen_corners,
    thirtytwo_corners,
    sixtyfour_corners,
)

df = pd.read_csv("dataset.csv")

df["coords_vals"] = df["coords_vals"].apply(eval)
df["sorted_coords"] = df["coords_vals"].apply(sort_coords)
df["interpolate"] = df["sorted_coords"].apply(interpolate)

df["edges4"] = df["interpolate"].apply(four_corners)
df["edges4"] = df["edges4"].apply(flatten)

df["edges8"] = df["interpolate"].apply(eight_corners)
df["edges8"] = df["edges8"].apply(flatten)

df["edges16"] = df["interpolate"].apply(sixteen_corners)
df["edges16"] = df["edges16"].apply(flatten)

df["edges32"] = df["interpolate"].apply(thirtytwo_corners)
df["edges32"] = df["edges32"].apply(flatten)

df["edges64"] = df["interpolate"].apply(sixtyfour_corners)
df["edges64"] = df["edges64"].apply(flatten)

df["bbox"] = df["sorted_coords"].apply(bbox)
df["center"] = df["sorted_coords"].apply(center)
# df["poly_area"] = df["interpolate"].apply(find_area)
# df["interpolate"] = df["interpolate"].apply(flatten)
# df["poly_area_percent"] = (df["poly_area"] / (256 * 256)) * 100
# df = df[(df["poly_area_percent"] <= 30)]

df["new4"] = df.apply(lambda x: np.concatenate((x["bbox"], x["center"], x["edges4"])), axis=1)
df["new4"] = df["new4"].apply(lambda x:list(x))

df["new8"] = df.apply(lambda x: np.concatenate((x["bbox"], x["center"], x["edges8"])), axis=1)
df["new8"] = df["new8"].apply(lambda x:list(x))

df["new16"] = df.apply(lambda x: np.concatenate((x["bbox"], x["center"], x["edges16"])), axis=1)
df["new16"] = df["new16"].apply(lambda x:list(x))

df["new32"] = df.apply(lambda x: np.concatenate((x["bbox"], x["center"], x["edges32"])), axis=1)
df["new32"] = df["new32"].apply(lambda x:list(x))

df["new64"] = df.apply(lambda x: np.concatenate((x["bbox"], x["center"], x["edges64"])), axis=1)
df["new64"] = df["new64"].apply(lambda x:list(x))

df.to_csv("dataset.csv", index=None)