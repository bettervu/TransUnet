import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from sqlalchemy.orm import joinedload
from dTurk.builders import model_builder
from bp.database import VisionClassifier, db_session
from dTurk.utils.helpers import prepare_and_slice_image, prepare_and_stitch_image

import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', message='WARNING:tensorflow:Value in checkpoint could not be found')

try:
    gpus = tf.config.list_physical_devices("GPU")
    tf.config.set_visible_devices(gpus[2], "GPU")
except:
    print("Gpus not found")

def custom_load(classifier_id, num_classes):
    with db_session() as sess:
        classifier_record = (
            sess.query(VisionClassifier)
            .options(joinedload(VisionClassifier.dataset))
            .filter(VisionClassifier.id == classifier_id)
            .first()
        )

    classifier_record.download()

    model = model_builder.build_model(
        model_name=classifier_record.model.split(":")[0],
        num_classes=num_classes,
        crop_width=256,
        crop_height=256,
        encoder_name=classifier_record.frontend,
        start_filters=16,
        encoder_weights=None,
    )

    model.load_weights(
        "/home/bv/dTurk/classifiers/"
        + classifier_record.name
        + "/model.ckpt/variables/variables".format(classifier_record.name)
    )

    return model, str(classifier_record.frontend)


def predict(image, model, name, num_classes, image_gsd=19):
    img = Image.open("juliuspred/" + image)
    img = np.array(img)
    if len(img.shape) > 2 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    c, d, e = prepare_and_slice_image(img, image_gsd, 10, 256, 0)
    c = model_builder.get_preprocessing(name)(c)
    p, q, h, w, r = c.shape
    c = c.reshape(p * q, h, w, r)
    prediction = model.predict(c)
    pred = prepare_and_stitch_image(prediction.reshape((p, q, h, w, num_classes)), image_gsd, 10, d, e, 256, 0)
    pred = pred * 255
    return img, pred


files = ["stadium.jpg"]
predict_options = {"base_layer": "Building", "depth": 0.001, "use_transformed_footprint": False}

classifiers = {
    "debris": 6801,
    "footprint": 4789,
    "hvac": 6560,
    "missing_shingles": 8649,
    "overhang": 7325,
    "patching": 8431,
    "ponding": 7399,
    "rust": 7621,
    "shingles": 6217,
    "solar_panels": 6234,
    "staining": 7092,
    "structural-damage": 6658,
    "swimming_pool": 7146,
    "tarp": 6924,
    "trampoline": 6528,
    "vegetation": 8058,
    "water-hazard": 7619,
    "yard-debris": 6594,
}

for classifier in classifiers:
    print(classifier)
    if (
        classifier == "missing_shingles"
        or classifier == "rust"
        or classifier == "shingles"
        or classifier == "solar_panels"
        or classifier == "vegetation"
        or classifier == "yard-debris"
    ):
        num_classes = 2
    else:
        num_classes = 3

    model, model_type = custom_load(classifiers[classifier], num_classes)
    for file in files:
        img, pred = predict(file, model, model_type, num_classes)

        if pred.shape[2] == 2:
            zeros = np.zeros_like(img)
            zeros[:,:,1] = pred[:,:,0]
            zeros[:,:,2] = pred[:,:,1]
            pred = zeros

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"juliuspred/{classifier}_img.png", img)

        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"juliuspred/{classifier}_pred.png", pred)

        print("\n\n\n\n")
