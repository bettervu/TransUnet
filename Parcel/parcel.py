import math
import operator
import os
import tarfile
from functools import reduce
from random import sample

import cv2
import gcsfs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, Permute, Reshape
from dTurk.models.SM_UNet import SM_UNet_Builder


FS = gcsfs.GCSFileSystem()
try:
    gpus = tf.config.list_physical_devices("GPU")
    tf.config.set_visible_devices(gpus[2], "GPU")
except:
    print("Gpus not found")


def extend_list(lol):
    if len(lol) >= 20:
        lol = sample(lol, 10)
    else:
        lol.extend((10 - len(lol)) * [[0, 0]])
    lol = np.array([(np.array(i).flatten()) for i in lol]).flatten()
    return lol


n_coords = 8


def interpolate(lol, n=200, t="linear"):
    if len(lol) == n:
        return lol
    elif len(lol) < n:
        final_x = []
        final_y = []
        x = [point[0] for point in lol]
        y = [point[1] for point in lol]
        x.append(x[0])
        y.append(y[0])
        n_to_inter_most = int(n / len(lol))
        n_to_inter_last = n % len(lol)
        for x_r in range(len(x) - 1):
            x1, y1, x2, y2 = x[x_r], y[x_r], x[x_r + 1], y[x_r + 1]
            if x_r == len(x) - 2:
                steps = np.linspace(0, 1, n_to_inter_most + n_to_inter_last + 1)
            else:
                steps = np.linspace(0, 1, n_to_inter_most + 1)
            if t == "same":
                current_x_to_interpolate = [x1] * len(steps)
                current_y_to_interpolate = [y1] * len(steps)
            else:
                current_x_to_interpolate = [(x1 + (x2 - x1) * (step)) for step in steps]
                current_y_to_interpolate = [(y1 + (y2 - y1) * (step)) for step in steps]
            final_x.extend(current_x_to_interpolate[:-1])
            final_y.extend(current_y_to_interpolate[:-1])
        lol = np.array([np.array([final_x[pt], final_y[pt]]) for pt in range(len(final_x))])
    return lol


def flatten(lol):
    lol = np.array([(np.array(i).flatten()) for i in lol]).flatten()
    return lol


def bbox(lol):
    x = [pt[0] for pt in lol]
    y = [pt[1] for pt in lol]
    return np.array([min(x), min(y), max(x), max(y)])



def center(lol):
    center = list(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), lol), [len(lol)] * 2))
    return np.array(center)


def find_area(coords):
    coords = np.rint(coords).astype(np.int32)
    img = cv2.fillPoly(np.zeros((256, 256)), [np.int32(coords)], (255, 0, 0))
    return len((np.where(img == 255))[0])


def load_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_png(byte_img)
    return img


def distance(l1, l2=[0, 0]):
    d = ((l1[0] - l2[0]) ** 2 + (l1[1] - l2[1]) ** 2) ** 0.5
    return d


def four_corners(lol):
    top_left_dst = list(map(lambda x: distance(x, [0,0]), lol))
    left_center_dst = list(map(lambda x: distance(x, [0, 128]), lol))
    bottom_left_dst = list(map(lambda x: distance(x, [0, 256]), lol))
    bottom_center_dst = list(map(lambda x: distance(x, [128, 256]), lol))
    bottom_right_dst = list(map(lambda x: distance(x, [256, 256]), lol))
    right_center_dst = list(map(lambda x: distance(x, [256, 128]), lol))
    top_right_dst = list(map(lambda x: distance(x, [256, 0]), lol))
    top_center_dst = list(map(lambda x: distance(x, [128, 0]), lol))
    top_left = lol[top_left_dst.index(min(top_left_dst))]
    left_center = lol[left_center_dst.index(min(left_center_dst))]
    bottom_left = lol[bottom_left_dst.index(min(bottom_left_dst))]
    bottom_center = lol[bottom_center_dst.index(min(bottom_center_dst))]
    bottom_right = lol[bottom_right_dst.index(min(bottom_right_dst))]
    right_center = lol[right_center_dst.index(min(right_center_dst))]
    top_right = lol[top_right_dst.index(min(top_right_dst))]
    top_center = lol[top_center_dst.index(min(top_center_dst))]
    return np.array([top_left, left_center, bottom_left, bottom_center, bottom_right, right_center, top_right, top_center])

def four_extremes(lol):
    x = [pt[0] for pt in lol]
    y = [pt[1] for pt in lol]
    left_most = lol[x.index(min(x))]
    bottom_most = lol[y.index(max(y))]
    right_most = lol[x.index(max(x))]
    top_most = lol[y.index(min(y))]
    return np.array([left_most, bottom_most, right_most, top_most])

def sort_coords(coords):
    dst = list(map(distance, coords))
    origin = dst.index(min(dst))
    final_coords = coords[origin:] + coords[:origin]
    return final_coords


df = pd.read_csv("dataset.csv")
df["coords_vals"] = df["coords_vals"].apply(eval)
# df = df[(df["after_cleanup_len"] <= n_coords)]
df["sorted_coords"] = df["coords_vals"].apply(sort_coords)
# df["edges"] = df["sorted_coords"].apply(four_corners)
df["edges"] = df["edges"].apply(flatten)
df["interpolate"] = df["sorted_coords"].apply(interpolate)
df["edges"] = df["interpolate"].apply(four_corners)
df["poly_area"] = df["interpolate"].apply(find_area)
df["interpolate"] = df["interpolate"].apply(flatten)
df["poly_area_percent"] = (df["poly_area"] / (256 * 256)) * 100
# df = df[(df["poly_area_percent"] <= 30)]
df["bbox"] = df["sorted_coords"].apply(bbox)
df["center"] = df["sorted_coords"].apply(center)
df["new"] = df.apply(lambda x: np.concatenate((x["bbox"], x["center"], x["edges"])), axis=1)
files = os.listdir("test_parcel/train")
try:
    files.remove(".DS_Store")
except:
    print("no hidden fle encountered")
files = [int(file.split(".")[0]) for file in files]
allowable_train_gtus = list(set(files).intersection(set(df["gtu_ids"])))
df = df[df["gtu_ids"].isin(allowable_train_gtus)]

images = []
missing = []
for i in df.index:
    try:
        img = cv2.imread(f"test_parcel/train/{df['gtu_ids'][i]}.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        if img.shape[2] == 4:
            missing.append(i)
    except:
        missing.append(i)
df.drop(missing, inplace=True)
df["images"] = images
print("No error until now")
X = df["images"].to_list()
X = np.array(X)
y = np.array(df["new"].to_list())

builder = SM_UNet_Builder(
    encoder_name="efficientnetv2-l",
    input_shape=(256, 256, 3),
    num_classes=3,
    activation="softmax",
    train_encoder=False,
    encoder_weights="imagenet",
    decoder_block_type="upsampling",
    head_dropout=0,  # dropout at head
    dropout=0,
)

model1 = builder.build_model()

model = Sequential(
    [
        Input(shape=(256, 256, 3)),
        model1,
        Conv2D((2 * n_coords) + 6, 2, 2),
        Flatten(),
        Dense(((2 * n_coords) + 6), activation="relu"),
    ]
)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, decay=0.0007)
loss = tf.keras.losses.MeanSquaredError()
model.compile(optimizer, loss)
callbacks = []
early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=20)

callbacks.append(early_stopping)

H = model.fit(
    np.asarray(X[:-125]),
    np.asarray(y[:-125]),
    validation_data=(X[-125:], y[-125:]),
    batch_size=16,
    epochs=10,
    verbose=1,
    callbacks=callbacks,
)

loss = H.history["loss"]
val_loss = H.history["val_loss"]
df = pd.DataFrame(loss)
df["loss"] = loss
df["val_loss"] = val_loss
df.to_csv("parcelUnet.csv")
model.save("my_model")

with tarfile.open("my_model.tar.gz", "w:gz") as tar:
    tar.add("my_model", arcname=os.path.basename("my_model"))