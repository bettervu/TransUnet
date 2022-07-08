import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import tensorflow as tf
from bp import Environment
from dTurk.utils.clr_callback import CyclicLR
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from train_helpers import dice_loss, mean_iou
import TransUnet.models.transunet as transunet
import TransUnet.experiments.config as conf
import masterful

env = Environment()

parser = argparse.ArgumentParser(description="masterful")
parser.add_argument("dataset")
parser.add_argument("gpu", type=int)
args, _ = parser.parse_known_args()
args_dict = vars(args)

config = conf.get_transunet()
config['image_size'] = 256
config["filters"] = 3
config['n_skip'] = 3
config['decoder_channels'] = [128, 64, 32, 16]
config['resnet']['n_layers'] = (3,4,9,12)
config['dropout'] = 0.1
config['grid'] = (28,28)
config["n_layers"] = 12

dataset = "MACH-77-it1"
machine = "local"
monitor = "val_loss"
epochs = 5
patience = 12
batch_size = 32
lr = 0.005
train_augmentation_file = "/Users/srinathramalingam/Desktop/codebase/dTurk/dTurk/augmentation/configs/light.yaml"
save_path = "weights/TU1"
checkpoint_filepath = save_path + "/checkpoint/"

dataset_directory = os.environ.get("BP_PATH_REMOTE") + "/datasets/semseg_base" + "/" + dataset

try:
    gpus = tf.config.list_physical_devices("GPU")
    tf.config.set_visible_devices(args_dict["gpu"], "GPU")
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


x_train = []
y_train = []
for i in range(len(train_input_names)):
    img = plt.imread(train_input_names[i])
    mask = plt.imread(train_label_names[i])
    x_train.append(img)
    y_train.append(mask)


dataset = tf.data.Dataset.from_tensor_slices((np.array(x_train), np.array(y_train)))

step_size = int(2.0 * len(train_input_names) / batch_size)
network = transunet.TransUnet(config, trainable=False)
network.model.compile(optimizer="adam", loss=dice_loss, metrics=mean_iou)
callbacks = []
cyclic_lr = CyclicLR(
    base_lr=lr / 10.0,
    max_lr=lr,
    step_size=step_size,
    mode="triangular2",
    cyclic_momentum=False,
    max_momentum=False,
    base_momentum=0.8,
)
callbacks.append(cyclic_lr)

early_stopping = EarlyStopping(
    monitor=monitor,
    mode="min" if "loss" in monitor else "max",
    patience=patience,
    verbose=1,
    restore_best_weights=True,
)
callbacks.append(early_stopping)

model_params = masterful.architecture.learn_architecture_params(
  model=network.model,
  task=masterful.enums.Task.SEMANTIC_SEGMENTATION,
  input_range=masterful.enums.ImageRange.ZERO_ONE,
  num_classes=3,
  prediction_logits=True,
)
training_dataset_params = masterful.data.learn_data_params(
  dataset=dataset,
  task=masterful.enums.Task.SEMANTIC_SEGMENTATION,
  image_range=masterful.enums.ImageRange.ZERO_ONE,
  num_classes=3,
  sparse_labels=False,
)


optimization_params = masterful.optimization.learn_optimization_params(
  network.model,
  model_params,
  dataset,
  training_dataset_params,
)

regularization_params = masterful.regularization.learn_regularization_params(
  network.model,
  model_params,
  optimization_params,
  dataset,
  training_dataset_params,
)

ssl_params = masterful.ssl.learn_ssl_params(
  dataset,
  training_dataset_params,
)


training_report = masterful.training.train(
  network.model,
  model_params,
  optimization_params,
  regularization_params,
  ssl_params,
  dataset,
  training_dataset_params,
)

print(training_report)