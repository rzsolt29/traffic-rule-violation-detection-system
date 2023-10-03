import torch
import os
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

# dataset loader
# Get the list of all files and directories
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
        anno['xmin'] = int(root.find("./object/bndbox/xmin").text)
        anno['ymin'] = int(root.find("./object/bndbox/ymin").text)
        anno['xmax'] = int(root.find("./object/bndbox/xmax").text)
        anno['ymax'] = int(root.find("./object/bndbox/ymax").text)

        if anno['class'] == "car_front" or anno['class'] == "truck_front":
            anno_list.append(anno)

    return pd.DataFrame(anno_list)


df_train = generate_train_df(anno_path)

# CNN model
