
from scr import config
import pandas as pd
import joblib
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


X_test = joblib.load("../" + config.TEST_IMAGES)
test_data = pd.read_csv("../" + config.TEST_DATA)


gender_model = load_model("models/gender_model.h5")
ethnicity_model = load_model("models/ethnicity.h5")
age_model = load_model("models/age_model.h5")

gender_model.evaluate(X_test, test_data.gender)
ethnicity_model.evaluate(X_test, test_data.ethnicity)
age_model.evaluate(X_test, test_data.age)



# preds = [np.argmax(i) for i in ethnicity_model.predict(X_test)]
# sns.heatmap(confusion_matrix(test_data.ethnicity, preds), annot=True, fmt="d")
# plt.show()
# print(test_data.shape)