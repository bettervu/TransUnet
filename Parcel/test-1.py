import gcsfs
import argparse
from tensorflow.keras.callbacks import EarlyStopping
from test import return_valid_dataset, get_gpu, create_dataset_semseg, build_unet, return_loss, return_metric, save_model


FS = gcsfs.GCSFileSystem()

def train_semseg(name, model_name, epochs, train_augmentation, val_augmentation, batch_size, crop_size, optimizer, loss, metric, gpu_no):

    get_gpu(gpu_no)

    df = return_valid_dataset(name)

    train, val = create_dataset_semseg(df, train_augmentation, val_augmentation, crop_size, batch_size)

    model = build_unet(crop_size)
    loss = return_loss(loss)
    metric = return_metric(metric)
    model.compile(optimizer, loss, metric)

    callbacks = []
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=4)
    callbacks.append(early_stopping)

    H = model.fit(
        train,
        validation_data=val,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
    )

    save_model(model_name, model)

parser = argparse.ArgumentParser(description="Train keypoint")
parser.add_argument("name", type=str)
parser.add_argument("model_name", type=str)
parser.add_argument("--train_augmentation", type=str, default=None)
parser.add_argument("--val_augmentation", type=str, default=None)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--crop_size", type=int, default=256)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--gpu_no", type=int, default=2)
parser.add_argument("--optimizer", type=str, default="adam")
parser.add_argument("--loss", type=str, default="mse")
parser.add_argument("--metric", type=str, default="iou")

args_dict = vars(parser.parse_args())

train_semseg(**args_dict)