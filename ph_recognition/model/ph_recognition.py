import pandas as pd
import pickle5 as pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

main_dir = "/Users/akshayokali/akshaynayanravi/my_workspace/ph_recognition"
data_dir = f"{main_dir}/ph_recognition/model/data"
raw_data_csv_file_path = f"{data_dir}/ph_recognition_dataset.csv"

data_raw = pd.read_csv(raw_data_csv_file_path)

# Binning Label Variable into Acid or Base
bins = [-1, 6, 7, 14]
labels = ["Acidic", "Neutral", "Basic"]
data_raw["type"] = pd.cut(data_raw["label"], bins, labels=labels)

# Dropping unnecessary variables
data = data_raw.drop(["label"], axis=1)

# Mapping Types into Numerical Values
type_mapping = {"Acidic": 0, "Neutral": 1, "Basic": 2}
data["type"] = data["type"].map(type_mapping)

# Describing Type as Targets
targets = data["type"]
# Describing other variables than class as Features
features = data.drop(["type"], axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    features, targets, test_size=0.2, random_state=420
)

random_forest_classifier = RandomForestClassifier()
random_forest_classifier.fit(x_train, y_train)
y_pred = random_forest_classifier.predict(x_test)

# Validating Model
acc_random_forest_classifier = round(accuracy_score(y_pred, y_test) * 100, 2)
pre_random_forest_classifier = round(
    precision_score(y_pred, y_test, average="macro") * 100, 2
)
rec_random_forest_classifier = round(
    recall_score(y_pred, y_test, average="macro") * 100, 2
)

print(
    f"Model Performance: \nAccuracy = {acc_random_forest_classifier} \nPrecision = {pre_random_forest_classifier} \nRecall = {rec_random_forest_classifier}"
)

model_pkl_file_path = f"{main_dir}/ph_recognition/model/random_forest_classifier.pkl"
pickle.dump(random_forest_classifier, open(model_pkl_file_path, "wb"))
