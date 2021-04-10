from scr import config
import pandas as pd
import numpy as np
import joblib

df = pd.read_csv(config.DATA_FOLDER)

def get_stratify_split(df, category):
    train_data = pd.DataFrame()
    val_data = pd.DataFrame()
    test_data = pd.DataFrame()

    for (label, group_img) in df.groupby(category):
        group_img = group_img.sample(len(group_img))

        train_data_end_index = int(len(group_img) * config.TRAIN_PCT)
        val_data_end_index = train_data_end_index + int(len(group_img) * config.VAL_PCT)

        train_data_in_group = group_img[:train_data_end_index]
        val_data_in_group = group_img[train_data_end_index:val_data_end_index]
        test_data_in_group = group_img[val_data_end_index:]

        print(len(train_data_in_group), len(val_data_in_group), len(test_data_in_group))

        train_data = train_data.append(train_data_in_group)
        val_data = val_data.append(val_data_in_group)
        test_data = test_data.append(test_data_in_group)
    return train_data, val_data, test_data


train_gender, val_gender, test_gender = get_stratify_split(df, "gender")
train_gender.shape, val_gender.shape, test_gender.shape


def pixel_preprocess(column):
    column = np.array([x for x in column.str.split()], dtype="float32")
    column = np.reshape(column, (-1, 48, 48, 1)) / 255
    return np.array(column)

X_train = pixel_preprocess(train_gender.pixels)
X_val = pixel_preprocess(val_gender.pixels)
X_test = pixel_preprocess(test_gender.pixels)


train_gender.to_csv(config.TRAIN_DATA)
val_gender.to_csv(config.VAL_DATA)
test_gender.to_csv(config.TEST_DATA)


joblib.dump(X_train, config.TRAIN_IMAGES)
joblib.dump(X_val, config.VAL_IMAGES)
joblib.dump(X_test, config.TEST_IMAGES)