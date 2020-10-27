import os
import random
import sys

import yaml
import numpy as np

with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    ymlfile.close()

if not cfg['use_gpu']:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

seed = cfg['seed']
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)

import tensorflow as tf
from tensorflow.keras.datasets.fashion_mnist import load_data
from tensorflow.keras.utils import to_categorical

tf.compat.v1.set_random_seed(seed)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.disable_eager_execution()  # To solve the speed problem of TF2

# Deprecated in tf2
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.intra_op_parallelism_threads = 4
config.inter_op_parallelism_threads = 4
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config))

from utils.deepnetwork import DeepNetwork
from utils.tracker import Tracker

def fashion_mnist(params):
    tracker = Tracker(seed, 'fashion_mnist.h5')

    # Load dataset
    (x_train, y_train), (x_test, y_test) = load_data()

    # Preprocessing
    # Reshape data as dataset is grayscaled
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    # Convert labels into categorial
    n_classes = params['n_classes']
    y_train = to_categorical(y_train, n_classes)
    y_test = to_categorical(y_test, n_classes)

    # Normalize images values
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # Create model
    model = DeepNetwork.build((28, 28, 1), params)

    # Train model
    model.fit(x_train, y_train,
        batch_size=params['batch_size'],
        epochs=params['n_epochs'],
        validation_data=(x_test, y_test),
        shuffle=True)

    # Evaluate performance
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    # Save model
    tracker.save_model(model)

if __name__ == "__main__":
    fashion_mnist(cfg['train'])