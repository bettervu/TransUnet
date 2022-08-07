import os
import tarfile
import gcsfs
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from helpers import load_image, mean_iou
from dTurk.models.SM_UNet import SM_UNet_Builder
from train_Unet import segmentation_loss

FS = gcsfs.GCSFileSystem()
try:
    gpus = tf.config.list_physical_devices("GPU")
    tf.config.set_visible_devices(gpus[1], "GPU")
except:
    print("Gpus not found")


files_x = os.listdir("test_parcel/train")
files_y = os.listdir("test_parcel/train_labels")

try:
    files_x.remove(".DS_Store")
    files_y.remove(".DS_Store")
except:
    print("No hidden file encountered")
files_x = [int(file.split(".")[0]) for file in files_x]
files_y = [int(file.split(".")[0]) for file in files_y]

allowable_files = list(set(files_x).intersection(set(files_y)))

df = pd.DataFrame()
df["gtu_ids"] = allowable_files

train_df = df.sample(frac=0.8)
val_df = df.drop(train_df.index)


train_x = tf.data.Dataset.from_tensor_slices(
    [f"test_parcel/train/{train_df['gtu_ids'][i]}.png" for i in train_df.index]
)
train_y = tf.data.Dataset.from_tensor_slices(
    [f"test_parcel/train_labels/{train_df['gtu_ids'][i]}.png" for i in train_df.index]
)

train_x = train_x.map(load_image)
train_x = train_x.map(lambda x: tf.ensure_shape(x, [256, 256, 3]))

train_y = train_y.map(load_image)
train_y = train_y.map(lambda x: tf.ensure_shape(x, [256, 256, 3]))

val_x = tf.data.Dataset.from_tensor_slices([f"test_parcel/train/{val_df['gtu_ids'][i]}.png" for i in val_df.index])
val_y = tf.data.Dataset.from_tensor_slices(
    [f"test_parcel/train_labels/{val_df['gtu_ids'][i]}.png" for i in val_df.index]
)

val_x = val_x.map(load_image)
val_x = val_x.map(lambda x: tf.ensure_shape(x, [256, 256, 3]))

val_y = val_y.map(load_image)
val_y = val_y.map(lambda x: tf.ensure_shape(x, [256, 256, 3]))


train = tf.data.Dataset.zip((train_x, train_y))
train = train.shuffle(5000)
train = train.batch(8)
train = train.prefetch(4)
val = tf.data.Dataset.zip((val_x, val_y))
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

model = builder.build_model()

model.compile(optimizer="adam", loss=segmentation_loss, metrics=mean_iou)

callbacks = []

early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=20)
callbacks.append(early_stopping)

H = model.fit(
    train,
    validation_data=(val),
    epochs=1,
    verbose=1,
    callbacks=callbacks,
)

iou = H.history["mean_iou"]
val_iou = H.history["val_mean_iou"]
loss = H.history["loss"]
val_loss = H.history["val_loss"]

df = pd.DataFrame(iou)

df.columns = ["mean_iou"]
df["val_mean_iou"] = val_iou
df["loss"] = loss
df["val_loss"] = val_loss

df.to_csv("parcel_Unet.csv")

model.save("my_model_Unet")

with tarfile.open("my_model_Unet.tar.gz", "w:gz") as tar:
    tar.add("my_model_Unet", arcname=os.path.basename("my_model_Unet"))
