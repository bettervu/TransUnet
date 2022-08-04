import os
import tarfile
import cv2
import gcsfs
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, Permute, Reshape

from helpers import interpolate, flatten, bbox, center, find_area, sort_coords, sixtyfour_corners
from dTurk.models.SM_UNet import SM_UNet_Builder

FS = gcsfs.GCSFileSystem()
try:
    gpus = tf.config.list_physical_devices("GPU")
    tf.config.set_visible_devices(gpus[2], "GPU")
except:
    print("Gpus not found")


n_coords = 64


df = pd.read_csv("dataset.csv")

df = df.head(100)
df["coords_vals"] = df["coords_vals"].apply(eval)
df["sorted_coords"] = df["coords_vals"].apply(sort_coords)
df["interpolate"] = df["sorted_coords"].apply(interpolate)
# df["edges"] = df["sorted_coords"].apply(four_corners)
df["edges"] = df["interpolate"].apply(sixtyfour_corners)
df["edges"] = df["edges"].apply(flatten)
df["bbox"] = df["sorted_coords"].apply(bbox)
df["center"] = df["sorted_coords"].apply(center)
print(len(df["edges"][0]))
print((df["edges"][0]))
print(len(df["edges"][0]))
# df["poly_area"] = df["interpolate"].apply(find_area)
# df["interpolate"] = df["interpolate"].apply(flatten)
# df["poly_area_percent"] = (df["poly_area"] / (256 * 256)) * 100
# df = df[(df["poly_area_percent"] <= 30)]

df["new"] = df.apply(lambda x: np.concatenate((x["bbox"], x["center"], x["edges"])), axis=1)
files = os.listdir("test_parcel/train")
try:
    files.remove(".DS_Store")
except:
    print("No hidden file encountered")
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
# y = np.array(df["bbox"].to_list())

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
        Conv2D((2 * n_coords) + 6 + 2, 2, 2),
        Flatten(),
        Dense(((2 * n_coords) + 6 + 2), activation="relu"),
    ]
)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, decay=0.0007)
loss = tf.keras.losses.MeanSquaredError()
model.compile(optimizer, loss)
callbacks = []
early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=20)
callbacks.append(early_stopping)
H = model.fit(
    np.asarray(X[:-500]),
    np.asarray(y[:-500]),
    validation_data=(X[-500:], y[-500:]),
    batch_size=12,
    epochs=4,
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
