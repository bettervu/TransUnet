import os
import tarfile
import gcsfs
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, Permute, Reshape

# from dTurk.generators.tf_data import TFDataBase

from helpers import load_image, convert_np

tf.config.run_functions_eagerly(True)


from dTurk.models.SM_UNet import SM_UNet_Builder

FS = gcsfs.GCSFileSystem()
try:
    gpus = tf.config.list_physical_devices("GPU")
    tf.config.set_visible_devices(gpus[2], "GPU")
except:
    print("Gpus not found")


n_coords = 32


df = pd.read_csv("dataset.csv")

df[f"new{n_coords}"] = df[f"new{n_coords}"].apply(eval)


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

print(len(train_df))

train_images = tf.data.Dataset.from_tensor_slices(
    [f"test_parcel/train/{train_df['gtu_ids'][i]}.png" for i in train_df.index]
)
train_images = train_images.map(load_image)
train_images = train_images.map(convert_np)
train_images = train_images.map(lambda x: tf.ensure_shape(x, [256, 256, 3]))
val_images = tf.data.Dataset.from_tensor_slices([f"test_parcel/train/{val_df['gtu_ids'][i]}.png" for i in val_df.index])
val_images = val_images.map(load_image)
val_images = val_images.map(convert_np)
val_images = val_images.map(lambda x: tf.ensure_shape(x, [256, 256, 3]))

y_train = np.array(train_df[f"new{n_coords}"].to_list())
y_val = np.array(val_df[f"new{n_coords}"].to_list())


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
        Dense(((2 * n_coords) + 6), activation="relu"),
    ]
)


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, decay=0.0007)
loss = tf.keras.losses.MeanSquaredError()
model.compile(optimizer, loss)
callbacks = []
early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=4)
callbacks.append(early_stopping)
H = model.fit(
    train,
    validation_data=(val),
    epochs=2,
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
