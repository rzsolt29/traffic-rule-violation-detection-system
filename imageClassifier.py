import os
import numpy as np
import cv2
import torch
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from sklearn.model_selection import train_test_split


device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

# dataset loader
# get the list of all files and directories
images_path = Path(r'D:\Dev\Szakdoga\datasets\training-1\jpg')
anno_path = Path(r'D:\Dev\Szakdoga\datasets\training-1\xml')


def filelist(root, file_type):
    """Returns a fully-qualified list of filenames under root directory"""
    return [os.path.join(directory_path, f) for directory_path, directory_name,
    files in os.walk(root) for f in files if f.endswith(file_type)]


def generate_train_df(anno_path):
    annotations = filelist(anno_path, '.xml')
    anno_list = []
    for anno_path in annotations:
        root = ET.parse(anno_path).getroot()
        anno = {}
        anno['filename'] = Path(str(images_path) + '/' + root.find("./filename").text)
        anno['width'] = root.find("./size/width").text
        anno['height'] = root.find("./size/height").text
        anno['class'] = root.find("./object/name").text
        left_col = int(root.find("./object/bndbox/xmin").text)
        anno['xmin'] = left_col
        bottom_row = int(root.find("./object/bndbox/ymin").text)
        anno['ymin'] = bottom_row
        right_col = int(root.find("./object/bndbox/xmax").text)
        anno['xmax'] = right_col
        top_row = int(root.find("./object/bndbox/ymax").text)
        anno['ymax'] = top_row
        anno['bb'] = np.array([left_col, top_row, right_col, bottom_row], dtype=np.float32)

        if anno['class'] == "car_front" or anno['class'] == "truck_front":
            anno_list.append(anno)

    return pd.DataFrame(anno_list)


df_train = generate_train_df(anno_path)


# label encode target
class_dict = {'car_front': 0, 'truck_front': 1}
df_train['class'] = df_train['class'].apply(lambda x: class_dict[x])

train_path_resized = Path(r'D:\Dev\Szakdoga\traffic-rule-violation-detection-system\dataset\images_resized')

#for i in df_train.index:
for i in range(0, 8):  # for test purposes we don't use the whole dataset
    img = cv2.imread(str(df_train['filename'][i]))
    cropped = img[int(df_train['ymin'][i]):int(df_train['ymax'][i]), int(df_train['xmin'][i]):int(df_train['xmax'][i])]
    resized = cv2.resize(cropped, (256, 256), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(train_path_resized, str(df_train['filename'][i])[40:]), resized)
    df_train.at[i, 'filename'] = os.path.join(train_path_resized, str(df_train['filename'][i])[40:])

df_train = df_train.reset_index()

X = df_train['filename']
Y = df_train['class']

# creating training and validation dataset
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# CNN model
