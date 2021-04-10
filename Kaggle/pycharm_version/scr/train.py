from scr import config
import pandas as pd
import joblib
import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPool2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import save_model

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


X_train = joblib.load("../" + config.TRAIN_IMAGES)
X_val = joblib.load("../" + config.VAL_IMAGES)
X_test = joblib.load("../" + config.TEST_IMAGES)

train_data = pd.read_csv("../" + config.TRAIN_DATA)
val_data = pd.read_csv("../" + config.VAL_DATA)
test_data = pd.read_csv("../" + config.TEST_DATA)

gender_model = Sequential([
    Conv2D(16, padding="same", kernel_size=(2, 2), input_shape=(config.IMG_SHAPE)),
    MaxPool2D((2, 2)),
    Conv2D(32, kernel_size=(3, 3), padding="same"),
    MaxPool2D((2, 2)),
    Conv2D(64, kernel_size=(3, 3), padding="valid"),
    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

gender_model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

gender_model_hist = gender_model.fit(X_train, train_data.gender, batch_size=64, epochs=8,
                                     validation_data=(X_val, val_data.gender))

print(gender_model.evaluate(X_test, test_data.gender))

gender_model.save(config.GENDER_MODEL + "gender_model.h5")


ethnicity_model = Model(inputs=gender_model.input, outputs=gender_model.layers[-2].output)
ethnicity_model = Sequential([
    ethnicity_model,
    Dense(len(set(train_data.ethnicity)), activation="softmax")
])

ethnicity_model.compile(optimizer=Adam(), loss="sparse_categorical_crossentropy", metrics=['accuracy'])

ethnicity_model_hist = ethnicity_model.fit(X_train, train_data.ethnicity,
                               batch_size=64, epochs=16,
                               validation_data=(X_val, val_data.ethnicity),
                               callbacks=[EarlyStopping(patience=2, restore_best_weights=True, monitor="val_accuracy")])


print(ethnicity_model.evaluate(X_test, test_data.ethnicity))
ethnicity_model.save_weights(config.GENDER_MODEL+"ethn_weights")
ethnicity_model.save(config.GENDER_MODEL + "ethnicity.h5")


age_prediciton_model = Sequential([
    Conv2D(16, padding="same", kernel_size=(2, 2), input_shape=(config.IMG_SHAPE)),
    MaxPool2D((2, 2)),
    Conv2D(32, kernel_size=(3, 3), padding="same"),
    Conv2D(32, kernel_size=(3, 3), padding="same"),
    MaxPool2D((2, 2)),
    Conv2D(64, kernel_size=(3, 3), padding="valid"),
    Conv2D(64, kernel_size=(3, 3), padding="valid"),
    BatchNormalization(),
    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="linear")
])

age_prediciton_model.compile(optimizer=Adam(), loss="mean_absolute_error", metrics=['mean_absolute_error'])

age_prediciton_model.fit(X_train, train_data.age,
                               batch_size=120, epochs=12,
                               validation_data=(X_val, val_data.age))

age_prediciton_model.evaluate(X_test, test_data.age)
age_prediciton_model.save(config.GENDER_MODEL + "age_model.h5")