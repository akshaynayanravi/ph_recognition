import numpy as np
import pickle5 as pickle

main_dir = "/Users/akshayokali/akshaynayanravi/my_workspace/ph_recognition"
model_pkl_file_path = f"{main_dir}/ph_recognition/model/random_forest_classifier.pkl"

type_mapping = {0: "Acidic", 1: "Neutral", 2: "Basic"}

ph_classifier = pickle.load(open(model_pkl_file_path, "rb"))


def get_ph_type(red: str, green: str, blue: str) -> str:
    features = [np.array([red, green, blue])]
    ph_value = ph_classifier.predict(features)

    ph_type = type_mapping[ph_value[0]]

    return ph_type


if __name__ == "__main__":
    red = "1"
    green = "240"
    blue = "140"

    ph_type = get_ph_type(red=red, green=green, blue=blue)
    print(ph_type)
