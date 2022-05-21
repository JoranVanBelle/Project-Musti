from email.policy import default
from gc import callbacks
import numpy as np
import os
import glob
import cv2
from requests import request
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split
from functools import partial

from flask import Flask, render_template, request

static = os.path.join("static")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = static

@app.route('/picture')
def index():

  numberOfFiles = len([name for name in os.listdir("./static/PicturesDemo")]) - 1

  picture = int(request.args.get("image", -1))

  if picture != -1:
    inp = os.path.join(app.config["UPLOAD_FOLDER"], "PicturesDemo")
    
    images = glob.glob(f"{inp}/*")

    X = []

    for i in images:
      img = cv2.imread(i)  # Original picture

      img = cv2.resize(img, (64,32))

      x = 1 - cv2.split(img)[0] / 255.0
      x = np.array(x).ravel().reshape(32, 64)

      X.append(x)

    X = np.array(X)

    model = keras.models.load_model("CNN.h5")

    prediction = model.predict(X[[[picture]]]).argmax(axis=1)

    zekerheid = model.predict(X[[[picture]]])
    zeker = zekerheid.tolist()[0][prediction[0]]

    

    map_dict = {0: "binnen", 1: "buiten", 2: "niet aanwezig"}

    loc = [map_dict[num] for num in prediction]
    print(loc)
    date = images[picture].split("\\")[-1].split(".")[0].split("_")[0]
    time = images[picture].split("\\")[-1].split(".")[0].split("_")[1]

    datum = f"{date[6:8]}/{date[4:6]}/{date[0:4]}"
    tijd = f"{time[0:2]}:{time[2:4]}"

    while zeker < 0.75 or loc[0] == "niet aanwezig":
      print("---------------------------------Announcement---------------------------------")
      print(f"The next picture had to be chosen, this one was not clear enough: {zeker}")
      print("---------------------------------Announcement---------------------------------")
      picture  += 1
      if picture > len(images) - 1:
        picture = 0
      inp = os.path.join(app.config["UPLOAD_FOLDER"], "PicturesDemo")
    
      images = glob.glob(f"{inp}/*")

      X = []

      for i in images:
        img = cv2.imread(i)  # Original picture

        img = cv2.resize(img, (64,32))

        x = 1 - cv2.split(img)[0] / 255.0
        x = np.array(x).ravel().reshape(32, 64)

        X.append(x)

      X = np.array(X)

      model = keras.models.load_model("CNN.h5")

      prediction = model.predict(X[[[picture]]]).argmax(axis=1)

      zekerheid = model.predict(X[[[picture]]])
      zeker = zekerheid.tolist()[0][prediction[0]]

      

      map_dict = {0: "binnen", 1: "buiten", 2: "niet aanwezig"}

      loc = [map_dict[num] for num in prediction]
      date = images[picture].split("\\")[-1].split(".")[0].split("_")[0]
      time = images[picture].split("\\")[-1].split(".")[0].split("_")[1]

      datum = f"{date[6:8]}/{date[4:6]}/{date[0:4]}"
      tijd = f"{time[0:2]}:{time[2:4]}"

    location = f"Musti was op {datum} om {tijd} {''.join(loc)} met {zeker*100:.2f}% zekerheid"
    return render_template('picture.html', picture = f"{images[picture]}", status=location, numberOfFiles=numberOfFiles, number=int(picture))
  else:
    return render_template('picture.html', picture = f"static/Layout images/Loading_icon.gif", status="Take a picture of Musti to see where she is")


@app.route('/retraining') 
def retraining():    

  def retrainModel():
    X = []
    y = []

    inside = os.path.join("input", "classificatie", "aanwezig")
    outside = os.path.join("input", "classificatie", "buiten")
    nothing = os.path.join("input", "classificatie", "niets")

    images = glob.glob(f"{inside}/*")

    for i in images:
        img = cv2.imread(i)
        img = cv2.resize(img, (64,32))

        x = 1 - cv2.split(img)[0] / 255.0
        x = np.array(x).ravel().reshape(32, 64)
        X.append(x)
        y.append(0)

    images = glob.glob(f"{outside}/*")

    for i in images:
        img = cv2.imread(i)
        img = cv2.resize(img, (64,32))

        x = 1 - cv2.split(img)[0] / 255.0
        x = np.array(x).ravel().reshape(32, 64)
        X.append(x)
        y.append(1)

    images = glob.glob(f"{nothing}/*")

    for i in images:
        img = cv2.imread(i)
        img = cv2.resize(img, (64,32))

        x = 1 - cv2.split(img)[0] / 255.0
        x = np.array(x).ravel().reshape(32, 64)
        X.append(x)
        y.append(2)
    
    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.15, random_state=42) 

    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

    DefaultConv2D = partial(keras.layers.Conv2D,
                            kernel_size=3, activation='relu', padding="SAME")

    model = keras.models.Sequential()
    model.add(DefaultConv2D(filters=64, kernel_size=7, input_shape=[32, 64, 1]))
    model.add(keras.layers.MaxPooling2D(pool_size=2))
    model.add(DefaultConv2D(filters=128))
    model.add(DefaultConv2D(filters=128))
    model.add(keras.layers.MaxPooling2D(pool_size=2))
    model.add(DefaultConv2D(filters=256))
    model.add(DefaultConv2D(filters=256))
    model.add(keras.layers.MaxPooling2D(pool_size=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(300, activation="relu"))
    model.add(keras.layers.Dense(150, activation="relu"))
    model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.Dense(3, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    
    callback = [keras.callbacks.ModelCheckpoint(
    filepath="CNN.h5",
    save_best_only=True,
    monitor="val_loss")
    ]

    model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid), verbose=0, callbacks=callback)

    test_model = keras.models.load_model("CNN.h5")
    y_pred = model.predict(X_test).argmax(axis=1)
    correct = 0
    for i, _ in enumerate(y_pred):
        if y_pred[i] == y_test[i]:
            correct+=1

    print("Test acc:", correct/len(y_pred))

  print("Starting to retrain the model")
  retrainModel()
  print("done retraining the model")

  return render_template("retraining.html")

if __name__ == '__main__':
  app.run(debug=True)
