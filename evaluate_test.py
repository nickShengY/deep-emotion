import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt

def evaluate_model(model, test):
    pre = Precision()
    re = Recall()
    acc = BinaryAccuracy()
    for batch in test.as_numpy_iterator(): 
        X, y = batch
        yhat = model.predict(X)
        pre.update_state(y, yhat)
        re.update_state(y, yhat)
        acc.update_state(y, yhat)
    print(pre.result(), re.result(), acc.result())

def predict_class(model, img_path):
    img = cv2.imread(img_path)
    plt.imshow(img)
    plt.show()
    resize = tf.image.resize(img, (256,256))
    plt.imshow(resize.numpy().astype(int))
    plt.show()
    yhat = model.predict(np.expand_dims(resize/255, 0))

    if yhat > 0.5: 
        print(f'Predicted class is Sad')
    else:
        print(f'Predicted class is Happy')

def save_model(model, model_path):
    model.save(model_path)

def load_model(model_path):
    return load_model(model_path)
