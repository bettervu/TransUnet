import math
import operator
import os
import tarfile
from functools import reduce
from random import sample

import cv2
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

def interpolate(lol, n=10):

    if len(lol) >= n:
        lol = sample(lol, n)
    else:
        final_x = []
        final_y = []
        x = [point[0] for point in lol]
        y = [point[1] for point in lol]
        x.append(x[0])
        y.append(y[0])
        n_to_inter_most = int(n/len(lol))
        n_to_inter_last = n%len(lol)
        for x_r in range(len(x)-1):
            x1,y1,x2,y2 = x[x_r],y[x_r],x[x_r+1],y[x_r+1]
            if x_r==len(x)-2:
                steps = np.linspace(0, 1, n_to_inter_most+n_to_inter_last+1)
            else:
                steps = np.linspace(0, 1, n_to_inter_most+1)
            current_x_to_interpolate = [(x1 + (x2-x1) * (step)) for step in steps]
            current_y_to_interpolate = [(y1 + (y2-y1) * (step)) for step in steps]
            final_x.extend(current_x_to_interpolate[:-1])
            final_y.extend(current_y_to_interpolate[:-1])
        lol = np.array([np.array([final_x[pt], final_y[pt]]) for pt in range(len(final_x))])
    return lol


def flatten(lol):
    lol = np.array([(np.array(i).flatten()) for i in lol]).flatten()
    return lol


df = pd.read_csv("dataset.csv")
df["coords_vals"] = df["coords_vals"].apply(eval)

images = []
for i in range(4665):
    img = cv2.imread(f"test_parcel/train/{i}.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(img)


def distance(l1, l2=[0, 0]):
    d = ((l1[0] - l2[0]) ** 2 + (l1[1] - l2[1]) ** 2) ** 0.5
    return d


def sort_coords(coords):
    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
    coords = (sorted(coords, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360))
    dst = list(map(distance, coords))
    origin = dst.index(min(dst))
    final_coords = coords[origin:] + coords[:origin]
    return final_coords


df["images"] = images
df = df[df["after_cleanup_len"] <= 10]
df["sorted_coords"] = df["coords_vals"].apply(sort_coords)
df["interpolate"] = df["sorted_coords"].apply(interpolate)
df["interpolate"] = df["interpolate"].apply(sort_coords)
df["interpolate"] = df["interpolate"].apply(flatten)

# df["coords_vals"]=df["coords_vals"].apply(extend_list)

X = df["images"].to_list()
X = [i / 255.0 for i in X]
X = np.array(X)
y = np.array(df["interpolate"].to_list())

model = Sequential([
    Input(shape=(256, 256, 3)),
    ResNet152V2(include_top=False, input_shape=(256, 256, 3)),
    Conv2D(512, 3, padding='same', activation='relu'),
    Conv2D(512, 3, padding='same', activation='relu'),
    Conv2D(256, 3, 2, padding='same', activation='relu'),
    Conv2D(256, 2, 2, activation='relu'),
    Dropout(0.05),
    Conv2D(20, 2, 2),
    Reshape((20,))
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, decay=0.0007)
loss = tf.keras.losses.MeanAbsoluteError()

model.compile(optimizer, loss)

callbacks = []
early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

callbacks.append(early_stopping)

H = model.fit(np.asarray(X[:-50]), np.asarray(y[:-50]), validation_data=(X[50:], y[50:]), batch_size=16, epochs=5, verbose=1)

loss = H.history["loss"]
val_loss = H.history["val_loss"]

df = pd.DataFrame(loss)
df["loss"] = loss
df["val_loss"] = val_loss

df.to_csv("parcelUnet.csv")

model.save("my_model")

with tarfile.open("my_model.tar.gz", "w:gz") as tar:
    tar.add("my_model", arcname=os.path.basename("my_model"))
