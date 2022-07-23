import math
import operator
import os
import tarfile
from functools import reduce
from random import sample

import gcsfs
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, Dropout, Input, Reshape

FS = gcsfs.GCSFileSystem()

try:
    gpus = tf.config.list_physical_devices("GPU")
    tf.config.set_visible_devices(gpus[2], "GPU")
except:
    print("Gpus not found")


def extend_list(lol):
    if len(lol) >= 10:
        lol = sample(lol, 10)
    else:
        lol.extend((10 - len(lol)) * [[0, 0]])
    lol = np.array([(np.array(i).flatten()) for i in lol]).flatten()
    return lol


def interpolate(lol, n=20, type="linear"):

    if len(lol) >= n:
        lol = sample(lol, n)
    else:
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
            if type == "same":
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


def distance(l1, l2=[0, 0]):
    d = ((l1[0] - l2[0]) ** 2 + (l1[1] - l2[1]) ** 2) ** 0.5
    return d


def sort_coords(coords):
    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
    coords = sorted(
        coords,
        key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360,
    )
    dst = list(map(distance, coords))
    origin = dst.index(min(dst))
    final_coords = coords[origin:] + coords[:origin]
    return final_coords


def load_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img


df = pd.read_csv("dataset.csv")
df["coords_vals"] = df["coords_vals"].apply(eval)

df = df[df["after_cleanup_len"] <= 20]
files = os.listdir("test_parcel/train")
files = [eval(file.split(".")[0]) for file in files]
allowable_train_gtus = list(set(files).intersection(set(df["gtu_ids"])))
df = df[df['gtu_ids'].isin(allowable_train_gtus)]
df["sorted_coords"] = df["coords_vals"].apply(sort_coords)
df["interpolate"] = df["sorted_coords"].apply(interpolate)
df["interpolate"] = df["interpolate"].apply(sort_coords)
df["interpolate"] = df["interpolate"].apply(flatten)

train_df = df.sample(frac=0.8)
val_df = df.drop(train_df.index)

train_images = tf.data.Dataset.from_tensor_slices([f"test_parcel/train/{train_df['gtu_ids'][i]}.png" for i in train_df.index])
train_images = train_images.map(load_image)
train_images = train_images.map(lambda x: tf.ensure_shape(x, [256, 256, 3]))

val_images = tf.data.Dataset.from_tensor_slices([f"test_parcel/train/{val_df['gtu_ids'][i]}.png" for i in val_df.index])
val_images = val_images.map(load_image)
val_images = val_images.map(lambda x: tf.ensure_shape(x, [256, 256, 3]))

y_train = np.array(train_df["interpolate"].to_list())
y_val = np.array(val_df["interpolate"].to_list())

train_labels = tf.data.Dataset.from_tensor_slices(y_train)
val_labels = tf.data.Dataset.from_tensor_slices(y_val)

train = tf.data.Dataset.zip((train_images, train_labels))
train = train.shuffle(5000)
train = train.batch(16)
train = train.prefetch(4)

val = tf.data.Dataset.zip((val_images, val_labels))
val = val.shuffle(1000)
val = val.batch(16)
val = val.prefetch(4)

model = Sequential(
    [
        Input(shape=(256, 256, 3)),
        ResNet152V2(include_top=False, input_shape=(256, 256, 3)),
        Conv2D(512, 3, padding="same", activation="relu"),
        Conv2D(512, 3, padding="same", activation="relu"),
        Conv2D(256, 3, 2, padding="same", activation="relu"),
        Conv2D(256, 2, 2, activation="relu"),
        Dropout(0.05),
        Conv2D(40, 2, 2),
        Reshape((40,)),
    ]
)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, decay=0.0007)
loss = tf.keras.losses.MeanAbsoluteError()

model.compile(optimizer, loss)

callbacks = []
early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=20)

callbacks.append(early_stopping)

H = model.fit(train, validation_data=val, epochs=20, verbose=1, callbacks=callbacks)


loss = H.history["loss"]
val_loss = H.history["val_loss"]

df = pd.DataFrame(loss)
df["loss"] = loss
df["val_loss"] = val_loss

df.to_csv("parcelUnet.csv")

model.save("my_model")

with tarfile.open("my_model.tar.gz", "w:gz") as tar:
    tar.add("my_model", arcname=os.path.basename("my_model"))
