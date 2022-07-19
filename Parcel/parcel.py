import cv2
import gcsfs
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.layers import Input,Conv2D,Dropout,Reshape

FS = gcsfs.GCSFileSystem()

try:
    gpus = tf.config.list_physical_devices("GPU")
    tf.config.set_visible_devices(gpus[2], "GPU")
except:
    print("Gpus not found")

def extend_list(lol):
    if len(lol) > 50:
        lol = lol[:50]
    else:
        lol.extend((50-len(lol))*[[0,0]])
    lol = np.array([(np.array(i).flatten())/(256) for i in lol]).flatten()
    return lol


df=pd.read_csv("dataset.csv")
df["coords_vals"]=df["coords_vals"].apply(eval)
df["coords_vals"]=df["coords_vals"].apply(extend_list)


images = []
for i in range(904):
    img = cv2.imread(f"test_parcel/train/{i}.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(img)


X = images
X = [i/255.0 for i in X]
X = np.array(X)
y = np.array(df["coords_vals"].to_list())

model = Sequential([
    Input(shape=(256,256,3)),
    ResNet152V2(include_top=False, input_shape=(256,256,3)),
    Conv2D(512, 3, padding='same', activation='relu'),
    Conv2D(512, 3, padding='same', activation='relu'),
    Conv2D(256, 3, 2, padding='same', activation='relu'),
    Conv2D(256, 2, 2, activation='relu'),
    Dropout(0.05),
    Conv2D(100, 2, 2),
    Reshape((100,))
])

model.compile(optimizer='adam',
             loss='mse',
             metrics=['accuracy'])


H = model.fit(np.asarray(X[:-100]), np.asarray(y[:-100]), validation_data=(X[100:], y[100:]), batch_size=16, epochs=100,verbose=1)

loss = H.history["loss"]
val_loss = H.history["val_loss"]

df = pd.DataFrame(loss)
df["loss"] = loss
df["val_loss"] = val_loss

df.to_csv("parcelUnet.csv")

model.save("my_model")