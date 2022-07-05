import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from bp import Environment
from dTurk.utils.clr_callback import CyclicLR
from tensorflow.keras.callbacks import EarlyStopping
from train_helpers import mean_iou, oversampling, create_dataset
from bp.database import db_session, Gtu
import cv2
from PIL import Image
from dTurk.models.SM_UNet import SM_UNet_Builder
from focal_loss import BinaryFocalLoss
import gcsfs
FS = gcsfs.GCSFileSystem()
env = Environment()

parser = argparse.ArgumentParser(description="Property")
parser.add_argument("gpu", type=int)
parser.add_argument("--machine", type=str, default="local")
parser.add_argument("--loss", type=str, default="iou")
parser.add_argument("--train_augmentation_file", type=str, default=None)
parser.add_argument("--val_augmentation_file", type=str, default=None)
parser.add_argument("--monitor", type=str, default="val_loss")
parser.add_argument("--lr", type=float, default=0.005)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--patience", type=int, default=6)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--save_path", type=str, default="unet")
parser.add_argument("--n_layers", type=int, default=12)

args, _ = parser.parse_known_args()
args_dict = vars(args)
args_dict["checkpoint_filepath"] = args_dict["save_path"] + "/checkpoint/"


try:
    gpus = tf.config.list_physical_devices("GPU")
    tf.config.set_visible_devices(gpus[args_dict["gpu"]], "GPU")
except:
    print("Gpus not found")


def possible_image_location(property_id, gtu_id):
    """deterines the ideal image location (whether or not inside gtu folder)"""
    gtu_id_dir = f"bvds/GTU/{property_id}/{gtu_id}/"
    property_id_dir = f"bvds/GTU/{property_id}/"
    return gtu_id_dir, property_id_dir

def open_image(gtu_id_dir: str, property_id_dir: str, reference_image: str):
    gtu_id_dir += reference_image
    property_id_dir += reference_image
    try:
        with FS.open(gtu_id_dir, "rb") as f:
            img = np.array(Image.open(f))
            return img
    except FileNotFoundError:
        try:
            with FS.open(property_id_dir, "rb") as f:
                img = np.array(Image.open(f))
                return img
        except FileNotFoundError:
            print("File not in location")

df = pd.read_csv("upload.csv")

img = []
prop = []

for gtu_id in df["gtu_ids"]:
    with db_session() as sess:
        ppid = sess.query(Gtu.property_id).filter(Gtu.id == gtu_id).first()[0]
    gtu_id_dir, property_id_dir = possible_image_location(property_id=ppid, gtu_id=gtu_id)
    image = open_image(gtu_id_dir, property_id_dir, "image.jpg")
    image = cv2.resize(image, (256,256))
#     image = image.reshape((1,256,256,3))
    image = tf.cast(image, tf.float32)
    image = image/255.0
    mask = open_image(gtu_id_dir, property_id_dir, "layer.Property.png")
    mask = cv2.resize(mask, (256,256))
#     mask = mask.reshape((1,256,256,1))
    mask = tf.cast(mask, tf.float32)
    mask = tf.expand_dims(mask, axis=-1)
    mask = mask/255.0
    img.append(image)
    prop.append(mask)

train = tf.data.Dataset.from_tensor_slices((np.array(img),np.array(prop)))
AT = tf.data.AUTOTUNE
BUFFER = 1000
STEPS_PER_EPOCH = 70//args_dict["batch_size"]
train = train.cache().shuffle(args_dict["batch_size"]).batch(args_dict["batch_size"])
train = train.prefetch(buffer_size=AT)

step_size = int(2.0 * len(img) / args_dict["batch_size"])

builder = SM_UNet_Builder(
    encoder_name='efficientnetv2-l',
    input_shape=(256, 256, 3),
    num_classes=1,
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
    # [b, h, w, classes]
    pred_tensor = tf.nn.softmax(y_pred)
    loss = 0.0
    for c in range(1):
        loss += dice_per_class(y_true[:, :, :, c], pred_tensor[:, :, :, c])
    return loss / 1

def segmentation_loss(y_true, y_pred):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    cross_entropy_loss = cce(y_true=y_true, y_pred=y_pred)
    dice_loss = gen_dice(y_true, y_pred)
    return 0.5 * cross_entropy_loss + 0.5 * dice_loss


model = builder.build_model()

model.compile(optimizer="adam", loss=segmentation_loss, metrics=mean_iou)

callbacks = []
# cyclic_lr = CyclicLR(
#     base_lr=args_dict["lr"] / 10.0,
#     max_lr=args_dict["lr"],
#     step_size=step_size,
#     mode="triangular2",
#     cyclic_momentum=False,
#     max_momentum=False,
#     base_momentum=0.8,
# )
# callbacks.append(cyclic_lr)

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
    train, epochs=args_dict["epochs"], callbacks=[callbacks]
)

iou = history.history["mean_iou"]
loss = history.history["loss"]

df = pd.DataFrame(iou)
df.columns = ["mean_iou"]
df["loss"] = loss

df.to_csv("property_unet_logs.csv")
