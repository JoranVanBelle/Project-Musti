import numpy as np
import os
import glob
import cv2
from requests import request
import seaborn as sns
import pandas as pd

import tensorflow as tf
from tensorflow import keras

from flask import Flask, render_template, request

MUSTI_FOLDER = os.path.join("static")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = MUSTI_FOLDER

@app.route('/')
def index():

  inp = os.path.join(app.config["UPLOAD_FOLDER"], "20220207_123451.jpg")

  X = []

  im = cv2.imread(inp)  # Original picture

  img = cv2.resize(im, (64,32))

  x = 1 - cv2.split(img)[0] / 255.0
  x = np.array(x).ravel().reshape(32, 64)

  X.append(x)

  X = np.asarray(X)

  model = keras.models.load_model("ANN.h5")

  prediction = model.predict(X).argmax(axis=1)

  zekerheid = model.predict(X)
  zeker = zekerheid.tolist()[0][prediction[0]]

  

  map_dict = {0: "binnen", 1: "buiten", 2: "niet aanwezig"}

  loc = [map_dict[num] for num in prediction]

  location = f"Musti is {''.join(loc)} met {zeker*100:.2f}% zekerheid"

  statuss = request.args.get("status", loc)

  return render_template('index.html', picture = inp, location=location)

if __name__ == '__main__':
  app.run(debug=True)
