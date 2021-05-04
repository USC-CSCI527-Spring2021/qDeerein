import numpy as np
import pytesseract as pt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers.core import Dense
from keras.models import model_from_json
from keras.optimizers import SGD
from matplotlib import pyplot as plt
 
from Dave import Dave
    
from train import train
from _test import _test

num_actions=6
def get_intial_cnn_model(grid_size, num_actions, hidden_size):
    model = keras.Sequential()
    model.add(Dense(hidden_size, input_shape=(grid_size,), activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(SGD(lr=.01), "mse")
    return model


def moving_average_diff(a, n=1):
    diff = np.diff(a)
    ret = np.cumsum(diff, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def load_model():
    # load json and create model
    json_file = open('model_epoch1000/Z1_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model_epoch1000/Z1_model.h5")
    loaded_model.compile(loss='mse', optimizer='sgd')
    return loaded_model


if __name__ == "__main__":
    grid_size=128
    hidden_size=512

    pt.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract'
    game = Dave()
    print("game object created")

    epoch =40
    num_of_games= 100
    train_mode = 1

    if train_mode == 1:
        # Train the model
        model = get_intial_cnn_model(grid_size,num_actions, hidden_size)
        hist,avg = train(game, model, epoch, num_of_games,num_actions,verbose=1)
        print("Training done")
    else:
        # Test the model
        model = load_model()
        hist = _test(game, model, epoch, num_of_games,num_actions,verbose=1)

    print(hist)
    np.savetxt('win_history.txt', hist)
    np.savetxt('average_points.txt', avg)

    f1=plt.figure()
    f2=plt.figure()

    a=f1.add_subplot(111)
    a.plot(moving_average_diff(hist))
    a.set_ylabel('average loss')
    a.set_xlabel('no of games')

    b=f2.add_subplot(111)
    b.plot(moving_average_diff(avg))
    b.set_ylabel('avg points per steps')
    b.set_xlabel('no of games')