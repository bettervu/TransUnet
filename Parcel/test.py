import os
import random
import tarfile
import numpy as np
import pandas as pd
import tensorflow as tf
from dTurk.models.SM_UNet import SM_UNet_Builder
from dTurk.generators import SemsegData
from dTurk.builders import model_builder
from dTurk.augmentation.transforms import get_train_transform_policy, get_validation_transform_policy
from dTurk.models.sm_models.losses import DiceLoss
from dTurk.metrics import WeightedMeanIoU
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, Permute, Reshape


ds_location = os.getenv("LOCAL_BP_PATH_REMOTE") + "/datasets/Parcel/"
md_location = os.getenv("LOCAL_BP_PATH_REMOTE") + "/dTurk/classifiers/"


def load_image(filename):
    img = tf.io.read_file(filename)
    image_decoded = tf.image.decode_png(img, channels=3)
    return image_decoded


def get_gpu(gpu_no):
    try:
        gpus = tf.config.list_physical_devices("GPU")
        tf.config.set_visible_devices(gpus[gpu_no], "GPU")
    except Exception as e:
        print(f"Gpus not found: {e}")


def remove_hidden_files(images):
    try:
        images.remove(".DS_Store")
    except Exception as e:
        print(f"No hidden file encountered: {e}")
    return images


def return_valid_dataset():
    images = os.listdir("test_parcel/train")
    images = remove_hidden_files(images)
    images = [int(image.split(".")[0]) for image in images]

    df = pd.read_csv("dataset.csv", index_col=None)
    allowable_gtus = list(set(images).intersection(set(df["gtu_ids"])))
    df = df[df["gtu_ids"].isin(allowable_gtus)]
    df["gtu_imgs"] = df["gtu_ids"].apply(lambda x: f"test_parcel/train/{x}.png")
    return df


def train_test_split(df):
    train_df = df.sample(frac=0.8)
    val_df = df.drop(train_df.index)
    return train_df, val_df

def load_images_df(df, crop_size):
    images = tf.data.Dataset.from_tensor_slices(df["gtu_imgs"].to_list())
    images = images.map(load_image)
    images = images.map(lambda x: tf.ensure_shape(x, [crop_size, crop_size, 3]))
    return images


def load_labels_df(df, n_coords):
    y = np.array(df[f"edges{n_coords}"].apply(eval).to_list())
    labels = tf.data.Dataset.from_tensor_slices(y)
    return labels


def create_batched_ds(images, labels, batch_size):
    ds = tf.data.Dataset.zip((images, labels))
    ds = ds.shuffle(random.randint(1000, 10000))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(4)
    return ds

def augment(subset, augmentation):
    if augmentation:
        if subset == "train":
            return get_train_transform_policy(augmentation_file=augmentation)
        return get_validation_transform_policy(augmentation_file=augmentation)
    return

def load_config(subset, augmentation):
    data = SemsegData(
        subset=subset,
        transform_policy=augment(subset, augmentation),
        preprocess=model_builder.get_preprocessing("s"),
        layer_colors=[[0, 0, 0], [255, 0, 0], [0, 255, 0]],
        use_mixup=False,
        use_sample_weights=False,
        use_distance_weights=False,
    )
    return data

def shapes_all(crop_size):
    image_shape = (crop_size, crop_size, 3)
    label_shape = (crop_size, crop_size, 3)
    shapes = [image_shape, label_shape]
    return shapes

def create_dataset_keypoint(df, n_coords, batch_size, crop_size):
    train_df, val_df = train_test_split(df)

    train_images = load_images_df(train_df, crop_size)
    val_images = load_images_df(val_df, crop_size)

    train_labels = load_labels_df(train_df, n_coords)
    val_labels = load_labels_df(val_df, n_coords)

    train = create_batched_ds(train_images, train_labels, batch_size)
    val = create_batched_ds(val_images, val_labels, batch_size)

    return train, val


def create_dataset_semseg(df, train_augmentation, val_augmentation, crop_size, batch_size):
    train_df, val_df = train_test_split(df)
    train_data = load_config("train", train_augmentation)
    val_data = load_config("val", val_augmentation)

    shapes = shapes_all(crop_size)

    train = train_data.get_tf_data(batch_size=batch_size, input_names=train_df["gtu_imgs"].to_list(), shapes=shapes)
    val = val_data.get_tf_data(batch_size=batch_size, input_names=val_df["gtu_imgs"].to_list(), shapes=shapes)

    return train, val


def build_unet(crop_size):
    builder = SM_UNet_Builder(
        encoder_name="efficientnetv2-l",
        input_shape=(crop_size, crop_size, 3),
        num_classes=3,
        activation="softmax",
        train_encoder=False,
        encoder_weights="imagenet",
        decoder_block_type="upsampling",
        head_dropout=0,
        dropout=0,
    )

    unet = builder.build_model()

    return unet


def build_keypoint(n_coords, crop_size):
    unet = build_unet(crop_size)

    keypoint = Sequential(
        [
            Input(shape=(crop_size, crop_size, 3)),
            unet,
            Conv2D((2 * n_coords) + 6 + 2, 2, 2),
            Flatten(),
            Dense(((2 * n_coords)), activation="relu"),
        ]
    )

    return keypoint


def return_loss(loss):
    if loss == "mse":
        loss = tf.keras.losses.MeanSquaredError()
    elif loss == "mae":
        loss = tf.keras.losses.MeanAbsoluteError()
    elif loss == "jaccard":
        loss = DiceLoss(class_weights=[1, 1, 1], class_indexes=[0, 1, 2], per_image=False)
    return loss

def return_metric(metric):
    if metric == "iou":
        metric = WeightedMeanIoU(num_classes=3, class_weights=[1, 1, 1], name="wt_mean_iou")
    return metric

def save_model(name, model):
    model.save(ds_location + '/' + name)

    with tarfile.open(ds_location + '/' + name+".tar.gz", "w:gz") as tar:
        tar.add(ds_location + '/' + name, arcname=os.path.basename(ds_location + '/' + name))


def get_callbacks(self, df, batch_size ):
    callbacks = []

    step_size = int(2.0 * len(df) / batch_size)
    callbacks.append(
        CyclicLR(
            base_lr=self.options["learning_rate"] / 10.0,
            max_lr=self.options["learning_rate"],
            step_size=step_size,
            mode=self.options.get("clr_mode", "triangular2"),
            cyclic_momentum=self.options["cyclic_momentum"],
            max_momentum=self.options["momentum"],
            base_momentum=self.options["momentum"] - 0.1,
        )
    )

    callbacks.append(
        EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=4)
    )

    callbacks.append(CSVLogger(os.path.join(self.log_path, "training.log"), append=True))

    callbacks.append(
        ModelCheckpoint(
            filepath=os.path.join(self.log_path, "model.ckpt"),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        )
    )

    callbacks.append(CustomTensorBoard(self.tensorboard_path, histogram_freq=1, update_freq=1))


    return callbacks