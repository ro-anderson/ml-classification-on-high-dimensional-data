import os
import pickle

class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class DataExtractor(metaclass=SingletonMeta):
    def __init__(self):
        self.data_path = "./data/mini_gm_public_v0.1.p"
        self.data = self._extract_data()
        self.data_exploration()

    def _extract_data(self):
        with open(self.data_path, 'rb') as f:
            data_dict = pickle.load(f)
        return data_dict

    def data_exploration(self):
        print("Type of data:", type(self.data))
        if isinstance(self.data, dict):
            print("Keys in data and their types:")
            for key, value in self.data.items():
                print(f"Key: {key}, Type: {type(value)}")

if __name__ == '__main__':
    de = DataExtractor()
