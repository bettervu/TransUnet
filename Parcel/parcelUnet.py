import os
import tarfile
import gcsfs
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from helpers import create_dataset

from dTurk.models.SM_UNet import SM_UNet_Builder
from dTurk.models.sm_models.losses import DiceLoss
from dTurk.metrics import WeightedMeanIoU

FS = gcsfs.GCSFileSystem()
try:
    gpus = tf.config.list_physical_devices("GPU")
    tf.config.set_visible_devices(gpus[2], "GPU")
except:
    print("Gpus not found")


images = os.listdir("test_parcel/train")
labels = os.listdir("test_parcel/train_labels")

try:
    images.remove(".DS_Store")
    labels.remove(".DS_Store")
except:
    print("No hidden file encountered")

images = [int(image.split(".")[0]) for image in images]
labels = [int(label.split(".")[0]) for label in labels]

allowable_gtus = list(set(images).intersection(set(labels)))

df = pd.DataFrame()
df["gtu_ids"] = allowable_gtus

train_df = df.sample(frac=0.8)
val_df = df.drop(train_df.index)

train_images = tf.data.Dataset.from_tensor_slices(
    [f"test_parcel/train/{train_df['gtu_ids'][i]}.png" for i in train_df.index]
)
# train_images = train_images.map(load_image)

train_labels = tf.data.Dataset.from_tensor_slices(
    [f"test_parcel/train_labels/{train_df['gtu_ids'][i]}.png" for i in train_df.index]
)
# train_labels = train_labels.map(load_image)

val_images = tf.data.Dataset.from_tensor_slices([f"test_parcel/train/{val_df['gtu_ids'][i]}.png" for i in val_df.index])
# val_images = val_images.map(load_image)

val_labels = tf.data.Dataset.from_tensor_slices([f"test_parcel/train_labels/{val_df['gtu_ids'][i]}.png" for i in val_df.index])
# val_labels = val_labels.map(load_image)

train, val = create_dataset(train_images, val_images, train_augmentation="dTurk/dTurk/augmentation/configs/light.yaml")


builder = SM_UNet_Builder(
    encoder_name='efficientnetv2-l',
    input_shape=(256, 256, 3),
    num_classes=3,
    activation="softmax",
    train_encoder=False,
    encoder_weights="imagenet",
    decoder_block_type="upsampling",
    head_dropout=0,  # dropout at head
    dropout=0,
)

loss = DiceLoss(class_weights=[1,1,1], class_indexes=[0,1,2], per_image=False)
metric = WeightedMeanIoU(
            num_classes=3, class_weights=[1,1,1], name="wt_mean_iou"
        )

model = builder.build_model()
model.compile(optimizer='adam', loss=loss, metrics=metric)

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

iou = H.history["wt_mean_iou"]
val_iou = H.history["val_wt_mean_iou"]
loss = H.history["loss"]
val_loss = H.history["val_loss"]

df = pd.DataFrame(loss)
df["loss"] = loss
df["val_loss"] = val_loss

df.to_csv("parcelUnet.csv")
model.save("my_model_Unet")
with tarfile.open("my_model_Unet.tar.gz", "w:gz") as tar:
    tar.add("my_model_Unet", arcname=os.path.basename("my_model_Unet"))


