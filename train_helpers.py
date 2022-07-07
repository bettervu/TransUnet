import os
import pickle
import imageio
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from dTurk.generators import SemsegData
from dTurk.builders import model_builder
from tensorflow.keras import backend as K
from dTurk.augmentation.transforms import get_train_transform_policy, get_validation_transform_policy


def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return 1 - numerator / denominator


def numpy_iou(y_true, y_pred, n_class=2):
    def iou(y_true, y_pred, n_class):
        IOU = []
        for c in range(n_class):
            TP = np.sum((y_true == c) & (y_pred == c))
            FP = np.sum((y_true != c) & (y_pred == c))
            FN = np.sum((y_true == c) & (y_pred != c))
            n = TP
            d = float(TP + FP + FN + 1e-12)
            iou = np.divide(n, d)
            IOU.append(iou)
        return np.mean(IOU)

    batch = y_true.shape[0]
    y_true = np.reshape(y_true, (batch, -1))
    y_pred = np.reshape(y_pred, (batch, -1))

    score = []
    for idx in range(batch):
        iou_value = iou(y_true[idx], y_pred[idx], n_class)
        score.append(iou_value)
    return np.mean(score)


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.5):
        y_pred_ = tf.cast(y_pred > t, tf.int32)
        score = tf.numpy_function(numpy_iou, [y_true, y_pred_], tf.float64)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def load_image(filename):
    data = np.empty((0,))
    data = imageio.imread(filename)
    if len(data.shape) == 3:
        if data.shape[0] in (3, 4):
            data = np.transpose(data, (1, 2, 0))
        if data.shape[2] == 4:
            data = data[:, :, :3]
    return data


def split_sample(layer_colored, malady_layer_dict):
    split_layer_name = []
    has_malady = False
    for layer_name, layer_color in malady_layer_dict.items():
        if np.all(np.equal(layer_colored, layer_color), axis=-1).any():
            split_layer_name.append(layer_name)
            has_malady = True
    if not has_malady:
        split_layer_name.append("base_layer")
    return split_layer_name


def _using_slice_tag(
    malady_layer_dict: dict, label_filename: str, machine: str, dataset: str, split_files, split_labels
):
    train_label_path = os.path.dirname(label_filename)
    with open(os.path.join(data_dir(machine, dataset=dataset), "semseg_slice_tag.pickle"), "rb") as slice_tag_file:
        while True:
            try:
                slice_label = pickle.load(slice_tag_file)
                for filename, metadata in slice_label.items():
                    for layer_name in split_sample(metadata["color"], malady_layer_dict):
                        split_files[layer_name].append(
                            os.path.join(train_label_path.replace("train_labels", "train"), filename)
                        )
                        split_labels[layer_name].append(os.path.join(train_label_path, filename))
            except EOFError:
                break


def data_dir(machine="akami", dataset="MACH-77-it1"):
    if machine == "local":
        dataset_directory = "/Users/srinathramalingam/mnt" + f"/datasets/semseg_base/{dataset}"
    else:
        dataset_directory = "/home/bv" + f"/datasets/semseg_base/{dataset}"
    return dataset_directory


def _using_slice_tag(
    malady_layer_dict: dict, label_filename: str, machine: str, dataset: str, split_files, split_labels
):
    train_label_path = os.path.dirname(label_filename)
    with open(os.path.join(data_dir(machine, dataset), "semseg_slice_tag.pickle"), "rb") as slice_tag_file:
        while True:
            try:
                slice_label = pickle.load(slice_tag_file)
                for filename, metadata in slice_label.items():
                    for layer_name in split_sample(metadata["color"], malady_layer_dict):
                        split_files[layer_name].append(
                            os.path.join(train_label_path.replace("train_labels", "train"), filename)
                        )
                        split_labels[layer_name].append(os.path.join(train_label_path, filename))
            except EOFError:
                break


def _using_local_files(
    malady_layer_dict: dict,
    train_labels,
    split_files,
    split_labels,
):
    for label_path in tqdm(train_labels):
        label_slice = load_image(label_path)
        for layer_name in split_sample(label_slice, malady_layer_dict):
            split_files[layer_name].append(label_path.replace("train_labels", "train"))
            split_labels[layer_name].append(label_path)


def _do_oversampling(train_labels, machine, dataset):
    malady_layer_dict = {"exposed_deck": [0, 255, 0]}

    split_files = {k: [] for k in list(malady_layer_dict.keys()) + ["base_layer"]}
    split_labels = {k: [] for k in list(malady_layer_dict.keys()) + ["base_layer"]}

    if os.path.exists(os.path.join(data_dir(machine, dataset=dataset), "semseg_slice_tag.pickle")):
        _using_slice_tag(
            malady_layer_dict,
            train_labels[0],
            machine=machine,
            dataset=dataset,
            split_files=split_files,
            split_labels=split_labels,
        )
    else:
        _using_local_files(malady_layer_dict, train_labels, split_files, split_labels)
    return split_files, split_labels


def oversampling(train_labels, machine, dataset: str, oversample: int):

    split_files, split_labels = _do_oversampling(train_labels, machine, dataset=dataset)
    major_class_counts = max(map(len, split_labels.values()))
    oversampling_factor = {
        k: round(major_class_counts / len(v))
        if oversample == -1
        else oversample
        if round(major_class_counts / len(v)) > 1
        else 1
        for k, v in split_files.items()
    }

    train_files = [item for k, v in split_files.items() for item in v * oversampling_factor[k]]
    train_files.sort()

    return train_files


def create_dataset(train_input_names, val_input_names, train_augmentation=None, val_augmentation=None):
    train_data = SemsegData(
        subset="train",
        transform_policy=get_train_transform_policy(augmentation_file=train_augmentation),
        preprocess=model_builder.get_preprocessing("s"),
        layer_colors=[[255, 255, 255]],
        use_mixup=False,
        use_sample_weights=False,
        use_distance_weights=False,
    )
    val_data = SemsegData(
        subset="val",
        transform_policy=val_augmentation,
        preprocess=model_builder.get_preprocessing("s"),
        layer_colors=[[255, 255, 255]],
        use_sample_weights=False,
        use_distance_weights=False,
    )

    image_shape = (256, 256, 3)
    label_shape = (
        256,
        256,
        None,
    )
    shapes = [image_shape, label_shape]

    train_ds_batched = train_data.get_tf_data(batch_size=12, input_names=train_input_names, shapes=shapes)
    val_ds_batched = val_data.get_tf_data(batch_size=12, input_names=val_input_names, shapes=shapes)

    return train_ds_batched, val_ds_batched
