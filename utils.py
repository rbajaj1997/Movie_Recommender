import time
import pickle
import os
import shutil


class ModelManager:
    path_name = ''

    @classmethod
    def __init__(cls, dataset_name=None, test_size=0.3):
        if not cls.path_name:
            cls.path_name = "model/" + dataset_name + '-testsize' + str(test_size)

    def save_model(self, model, save_name = str):
        if 'pkl' not in save_name:
            save_name += '.pkl'
        if not os.path.exists('model'):
            os.mkdir('model')
        pickle.dump(model, open(self.path_name + "-%s" % save_name, "wb"), protocol=2)

    def load_model(self, model_name= str):
        if 'pkl' not in model_name:
            model_name += '.pkl'
        if not os.path.exists(self.path_name + "-%s" % model_name):
            raise OSError('There is no model named %s in model/ dir' % model_name)
        return pickle.load(open(self.path_name + "-%s" % model_name, "rb"))

