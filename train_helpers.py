import os
import pickle
import imageio
import numpy as np
from tqdm import tqdm
from dTurk.generators import SemsegData
from dTurk.builders import model_builder
from dTurk.metrics import MeanIoU, WeightedMeanIoU
from dTurk.augmentation.transforms import get_train_transform_policy, get_validation_transform_policy
from dTurk.models.sm_models.losses import CategoricalCELoss, CategoricalFocalLoss, DiceLoss, JaccardLoss


def get_loss(loss_name: str):
    name = loss_name.lower()

    class_weights = [1, 1, 1]
    class_indexes = [0, 1, 2]

    if name == "weighted_categorical_cross_entropy":
        loss_function = CategoricalCELoss(class_weights=class_weights, class_indexes=class_indexes)
    elif name in ["iou", "jaccard"]:
        loss_function = JaccardLoss(class_weights=class_weights, class_indexes=class_indexes, per_image=False)
    elif name in ["dice", "f1"]:
        loss_function = DiceLoss(class_weights=class_weights, class_indexes=class_indexes, per_image=False)
    elif name in ["focal_dice"]:
        dice_loss = DiceLoss(class_weights=class_weights, class_indexes=class_indexes, per_image=False)
        focal_loss = CategoricalFocalLoss(alpha=0.25, gamma=2, class_indexes=class_indexes)
        loss_function = dice_loss + focal_loss
    elif name in ["focal_iou"]:
        iou_loss = JaccardLoss(class_weights=class_weights, class_indexes=class_indexes, per_image=False)
        focal_loss = CategoricalFocalLoss(alpha=0.25, gamma=2, class_indexes=class_indexes)
        loss_function = iou_loss + focal_loss
    return loss_function

def iou():
    class_weights_primary = np.array([0,0,1])
    class_weights_primary = class_weights_primary / class_weights_primary.sum()

    metric =  WeightedMeanIoU(
            num_classes=3, class_weights=class_weights_primary, name="primary_mean_iou"
        )
    return metric


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
        if (np.all(np.equal(layer_colored, layer_color), axis=-1).any()):
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


def data_dir(machine='akami', dataset='MACH-77-it1'):
    if machine == 'local':
        dataset_directory='/Users/srinathramalingam/mnt' + f'/datasets/semseg_base/{dataset}'
    else:
        dataset_directory='/home/bv' +  f'/datasets/semseg_base/{dataset}'
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
            split_files[layer_name].append(
                label_path.replace("train_labels", "train")
            )
            split_labels[layer_name].append(label_path)


def _do_oversampling(train_labels, machine, dataset):
    malady_layer_dict = {'exposed_deck': [0, 255, 0]}

    split_files = {k: [] for k in list(malady_layer_dict.keys()) + ["base_layer"]}
    split_labels = {k: [] for k in list(malady_layer_dict.keys()) + ["base_layer"]}

    if os.path.exists(os.path.join(data_dir(machine, dataset=dataset), "semseg_slice_tag.pickle")):
        _using_slice_tag(malady_layer_dict, train_labels, machine=machine, dataset=dataset, split_files=split_files, split_labels=split_labels)
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
        preprocess=model_builder.get_preprocessing('s'),
        layer_colors=[[0, 0, 0], [255, 0, 0], [0, 255, 0]],
        use_mixup=False,
        use_sample_weights=False,
        use_distance_weights=False,
    )
    val_data = SemsegData(
        subset="val",
        transform_policy=val_augmentation,
        preprocess=model_builder.get_preprocessing('s'),
        layer_colors=[[0, 0, 0], [255, 0, 0], [0, 255, 0]],
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

    train_ds_batched = train_data.get_tf_data(
        batch_size=12, input_names=train_input_names, shapes=shapes
    )
    val_ds_batched = val_data.get_tf_data(
        batch_size=12, input_names=val_input_names, shapes=shapes
    )

    return train_ds_batched, val_ds_batched