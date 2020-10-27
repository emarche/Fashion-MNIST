import yaml
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model

with open('config.yml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    ymlfile.close()
seed = cfg['seed']
tf.compat.v1.set_random_seed(seed)

class DeepNetwork:
    @staticmethod  
    def build(input_shape, params):

        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(0.01), input_shape=input_shape, activation='relu'))
        model.add(Conv2D(32, (5, 5), kernel_regularizer=l2(0.01), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.01), activation='relu'))
        model.add(Conv2D(64, (5, 5), kernel_regularizer=l2(0.01), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(params['n_classes'], activation='softmax'))

        model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
              metrics=['accuracy'])

        if params['print_model']:
            plot_model(model, to_file='model.png', show_shapes=True)    # Creates a png with the network architecture

        return model

    @staticmethod  
    def print_weights(model):
        print(model.summary())
        print("Configuration: " + str(model.get_config()))
        for layer in model.layers:
            print(layer.name)
            print(layer.get_weights())  #   Returns weights of the layer (layer.get_weights()[0]) and bias ([1])

