import os
import numpy as np

class Tracker:
    def __init__(self, seed, model_name):
        self.save_tag = model_name + '_seed_' + str(seed)

        self.model_save = "models/"
        if not os.path.exists(self.model_save): os.makedirs(self.model_save)

    def save_model(self, model):
        model.save(self.model_save + self.save_tag + '.h5')