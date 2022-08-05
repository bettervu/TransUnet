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

from helpers import (
    interpolate,
    flatten,
    bbox,
    center,
    find_area,
    sort_coords,
    load_image,
    four_corners,
    eight_corners,
    sixteen_corners,
    thirtytwo_corners,
    sixtyfour_corners,
)
from dTurk.models.SM_UNet import SM_UNet_Builder

FS = gcsfs.GCSFileSystem()
try:
    gpus = tf.config.list_physical_devices("GPU")
    tf.config.set_visible_devices(gpus[1], "GPU")
except:
    print("Gpus not found")


n_coords = 32


df = pd.read_csv("dataset.csv")

# df["coords_vals"] = df["coords_vals"].apply(eval)
# df["sorted_coords"] = df["coords_vals"].apply(sort_coords)
# df["interpolate"] = df["sorted_coords"].apply(interpolate)
#
# df["edges4"] = df["interpolate"].apply(four_corners)
# df["edges4"] = df["edges4"].apply(flatten)
#
# df["edges8"] = df["interpolate"].apply(eight_corners)
# df["edges8"] = df["edges8"].apply(flatten)
#
# df["edges16"] = df["interpolate"].apply(sixteen_corners)
# df["edges16"] = df["edges16"].apply(flatten)
#
# df["edges32"] = df["interpolate"].apply(thirtytwo_corners)
# df["edges32"] = df["edges32"].apply(flatten)
#
# df["edges64"] = df["interpolate"].apply(sixtyfour_corners)
# df["edges64"] = df["edges64"].apply(flatten)
#
# df["bbox"] = df["sorted_coords"].apply(bbox)
# df["center"] = df["sorted_coords"].apply(center)
# # df["poly_area"] = df["interpolate"].apply(find_area)
# # df["interpolate"] = df["interpolate"].apply(flatten)
# # df["poly_area_percent"] = (df["poly_area"] / (256 * 256)) * 100
# # df = df[(df["poly_area_percent"] <= 30)]
#
# df["new4"] = df.apply(lambda x: np.concatenate((x["bbox"], x["center"], x["edges4"])), axis=1)
# df["new4"] = df["new4"].apply(lambda x:list(x))
#
# df["new8"] = df.apply(lambda x: np.concatenate((x["bbox"], x["center"], x["edges8"])), axis=1)
# df["new8"] = df["new8"].apply(lambda x:list(x))
#
# df["new16"] = df.apply(lambda x: np.concatenate((x["bbox"], x["center"], x["edges16"])), axis=1)
# df["new16"] = df["new16"].apply(lambda x:list(x))
#
# df["new32"] = df.apply(lambda x: np.concatenate((x["bbox"], x["center"], x["edges32"])), axis=1)
# df["new32"] = df["new32"].apply(lambda x:list(x))
#
# df["new64"] = df.apply(lambda x: np.concatenate((x["bbox"], x["center"], x["edges64"])), axis=1)
# df["new64"] = df["new64"].apply(lambda x:list(x))
#
# df.to_csv("dataset.csv")

df["new32"] = df["new32"].apply(eval)


files = os.listdir("test_parcel/train")
try:
    files.remove(".DS_Store")
except:
    print("No hidden file encountered")
files = [int(file.split(".")[0]) for file in files]
allowable_train_gtus = list(set(files).intersection(set(df["gtu_ids"])))
df = df[df["gtu_ids"].isin(allowable_train_gtus)]

train_df = df.sample(frac=0.8)
val_df = df.drop(train_df.index)


train_images = tf.data.Dataset.from_tensor_slices(
    [f"test_parcel/train/{train_df['gtu_ids'][i]}.png" for i in train_df.index]
)
train_images = train_images.map(load_image)
train_images = train_images.map(lambda x: tf.ensure_shape(x, [256, 256, 3]))
val_images = tf.data.Dataset.from_tensor_slices([f"test_parcel/train/{val_df['gtu_ids'][i]}.png" for i in val_df.index])
val_images = val_images.map(load_image)
val_images = val_images.map(lambda x: tf.ensure_shape(x, [256, 256, 3]))

y_train = np.array(train_df["new32"].to_list())
y_val = np.array(val_df["new32"].to_list())


train_labels = tf.data.Dataset.from_tensor_slices(y_train)
val_labels = tf.data.Dataset.from_tensor_slices(y_val)

train = tf.data.Dataset.zip((train_images, train_labels))
train = train.shuffle(5000)
train = train.batch(8)
train = train.prefetch(4)
val = tf.data.Dataset.zip((val_images, val_labels))
val = val.shuffle(1000)
val = val.batch(8)
val = val.prefetch(4)

print("No error until now")


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
    train,
    validation_data=(val),
    epochs=3,
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
