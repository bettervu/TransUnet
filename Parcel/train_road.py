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


train_images = ["Dataset_mod/train/" + i for i in os.listdir("Dataset_mod/train")]
val_images = ["Dataset_mod/val/" + i for i in os.listdir("Dataset_mod/val")]

train, val = create_dataset(train_images, val_images, train_augmentation="dTurk/dTurk/augmentation/configs/light.yaml")


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

loss = DiceLoss(class_weights=[1, 1, 0], class_indexes=[0, 1, 2], per_image=False)
metric = WeightedMeanIoU(num_classes=3, class_weights=[1, 1, 0], name="wt_mean_iou")

model = builder.build_model()
model.compile(optimizer="adam", loss=loss, metrics=metric)

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

df.to_csv("roadUnet.csv")
model.save("road_model_Unet")
with tarfile.open("road_model_Unet.tar.gz", "w:gz") as tar:
    tar.add("road_model_Unet", arcname=os.path.basename("road_model_Unet"))