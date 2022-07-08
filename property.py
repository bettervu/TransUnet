import cv2
import numpy as np
import os
import argparse
import pandas as pd
import tensorflow as tf
from bp import Environment
from dTurk.utils.clr_callback import CyclicLR
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from train_helpers import mean_iou, create_dataset, dice_loss
from dTurk.models.SM_UNet import SM_UNet_Builder
from focal_loss import BinaryFocalLoss
import gcsfs
FS = gcsfs.GCSFileSystem()

import TransUnet.models.transunet as transunet
import TransUnet.experiments.config as conf

parser = argparse.ArgumentParser(description="Property")
parser.add_argument("dataset")
parser.add_argument("log", type=str)
parser.add_argument("gpu", type=int)
parser.add_argument("--machine", type=str, default="local")
parser.add_argument("--loss", type=str, default="iou")
parser.add_argument("--train_augmentation_file", type=str, default=None)
parser.add_argument("--val_augmentation_file", type=str, default=None)
parser.add_argument("--monitor", type=str, default="val_loss")
parser.add_argument("--lr", type=float, default=0.005)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--patience", type=int, default=20)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--save_path", type=str, default="property")

args, _ = parser.parse_known_args()
args_dict = vars(args)
args_dict["checkpoint_filepath"] = args_dict["save_path"] + "/checkpoint/"

if args_dict["machine"] == "local":
    dataset_directory = os.environ.get("BP_PATH_REMOTE") + "/datasets/semseg_base" + "/" + args_dict["dataset"]
else:
    dataset_directory = "/home/bv/" + "datasets/semseg_base" + "/" + args_dict["dataset"]

try:
    gpus = tf.config.list_physical_devices("GPU")
    tf.config.set_visible_devices(gpus[args_dict["gpu"]], "GPU")
except:
    print("Gpus not found")

train_input_names = [
    dataset_directory + "/train/" + i
    for i in os.listdir(dataset_directory + "/train/")
    if i.endswith(".png")
]
train_label_names = [
    dataset_directory + "/train_labels/" + i
    for i in os.listdir(dataset_directory + "/train/")
    if i.endswith(".png")
]
val_input_names = [
    dataset_directory + "/val/" + i for i in os.listdir(dataset_directory + "/val/") if i.endswith(".png")
]
val_label_names = [
    dataset_directory + "/val_labels/" + i for i in os.listdir(dataset_directory + "/val/") if i.endswith(".png")
]

val_input_names = [
    dataset_directory + "/val/" + i for i in os.listdir(dataset_directory + "/val/") if i.endswith(".png")
]
val_label_names = [
    dataset_directory + "/val_labels/" + i for i in os.listdir(dataset_directory + "/val/") if i.endswith(".png")
]

x_train = []
y_train = []
for i in range(len(train_input_names)):
    img = cv2.imread(train_input_names[i])
    mask = cv2.imread(train_label_names[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask = mask.astype(np.float32)
    x_train.append(img)
    y_train.append(mask)

x_val = []
y_val = []
for i in range(len(val_input_names)):
    img = cv2.imread(val_input_names[i])
    mask = cv2.imread(val_label_names[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask = mask.astype(np.float32)
    x_val.append(img)
    y_val.append(mask)

train_ds = tf.data.Dataset.from_tensor_slices((np.array(x_train), np.array(y_train))).batch(16, drop_remainder=True)
val_ds = tf.data.Dataset.from_tensor_slices((np.array(x_val), np.array(y_val))).batch(16, drop_remainder=True)

# train_ds_batched, val_ds_batched = create_dataset(train_input_names, val_input_names, train_augmentation=args_dict["train_augmentation_file"])

builder = SM_UNet_Builder(
    encoder_name='efficientnetv2-l',
    input_shape=(256, 256, 3),
    num_classes=3,
    activation="softmax",
    train_encoder=False,
    encoder_weights="imagenet",
    decoder_block_type="upsampling",
    head_dropout=0,  # dropout at head
    dropout=0,  # dropout at feature extraction
)

def dice_per_class(y_true, y_pred, eps=1e-5):
    intersect = tf.reduce_sum(y_true * y_pred)
    y_sum = tf.reduce_sum(y_true * y_true)
    z_sum = tf.reduce_sum(y_pred * y_pred)
    loss = 1 - (2 * intersect + eps) / (z_sum + y_sum + eps)
    return loss

def gen_dice(y_true, y_pred):
    """both tensors are [b, h, w, classes] and y_pred is in logit form"""
    pred_tensor = tf.nn.softmax(y_pred)
    loss = 0.0
    for c in range(3):
        loss += dice_per_class(y_true[:, :, :, c], pred_tensor[:, :, :, c])
    return loss / 3

def segmentation_loss(y_true, y_pred):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    cross_entropy_loss = cce(y_true=y_true, y_pred=y_pred)
    dice_loss = gen_dice(y_true, y_pred)
    return 0.5 * cross_entropy_loss + 0.5 * dice_loss

model = builder.build_model()
model.compile(optimizer='adam', loss=segmentation_loss, metrics=mean_iou)

step_size = int(2.0 * len(train_input_names) / args_dict["batch_size"])
callbacks = []
cyclic_lr = CyclicLR(
    base_lr=args_dict["lr"] / 10.0,
    max_lr=args_dict["lr"],
    step_size=step_size,
    mode="triangular2",
    cyclic_momentum=False,
    max_momentum=False,
    base_momentum=0.8,
)
callbacks.append(cyclic_lr)

early_stopping = EarlyStopping(
    monitor=args_dict["monitor"],
    mode="min" if "loss" in args_dict["monitor"] else "max",
    patience=args_dict["patience"],
    verbose=1,
    restore_best_weights=True,
)
callbacks.append(early_stopping)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=args_dict["checkpoint_filepath"],
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True)
callbacks.append(cp_callback)



history = model.fit(
    train_ds, epochs=100, validation_data=val_ds, callbacks=[callbacks]
)

iou = history.history["mean_iou"]
val_iou = history.history["val_mean_iou"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

df = pd.DataFrame(loss)
df["loss"] = loss
df["val_loss"] = val_loss

df.to_csv("parcelUnet.csv")

model.save("my_model")