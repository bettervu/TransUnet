import os
import cv2
import gcsfs
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from bp.database import db_session, Gtu
from tqdm import tqdm
FS = gcsfs.GCSFileSystem()


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


os.makedirs("Dataset", exist_ok=True)
os.makedirs("Dataset/train", exist_ok=True)
os.makedirs("Dataset/train_labels", exist_ok=True)
os.makedirs("Dataset/val", exist_ok=True)
os.makedirs("Dataset/val_labels", exist_ok=True)

length = len(df)


for gtu_id in tqdm(df["gtu_ids"][:int(0.8*length)]):
    with db_session() as sess:
        ppid = sess.query(Gtu.property_id).filter(Gtu.id == gtu_id).first()[0]
    gtu_id_dir, property_id_dir = possible_image_location(property_id=ppid, gtu_id=gtu_id)
    try:
        image = open_image(gtu_id_dir, property_id_dir, "image.jpg")
        image = cv2.resize(image, (256,256))
        mask = open_image(gtu_id_dir, property_id_dir, "layer.Property.png")
        mask = cv2.resize(mask, (256,256))
        plt.imsave(f"Dataset/train/{gtu_id}.png", image)
        cv2.imwrite(f"Dataset/train_labels/{gtu_id}.png", mask)
    except:
        print(gtu_id)


for gtu_id in tqdm(df["gtu_ids"][int(0.8*length):]):
    with db_session() as sess:
        ppid = sess.query(Gtu.property_id).filter(Gtu.id == gtu_id).first()[0]
    gtu_id_dir, property_id_dir = possible_image_location(property_id=ppid, gtu_id=gtu_id)
    try:
        image = open_image(gtu_id_dir, property_id_dir, "image.jpg")
        image = cv2.resize(image, (256,256))
        mask = open_image(gtu_id_dir, property_id_dir, "layer.Property.png")
        mask = cv2.resize(mask, (256,256))
        plt.imsave(f"Dataset/val/{gtu_id}.png", image)
        cv2.imwrite(f"Dataset/val_labels/{gtu_id}.png", mask)
    except:
        print(gtu_id)