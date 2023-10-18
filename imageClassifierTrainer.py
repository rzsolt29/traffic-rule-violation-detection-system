import os
import numpy as np
import cv2
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


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

# image crop and resize
train_path_resized = Path(r'D:\Dev\Szakdoga\traffic-rule-violation-detection-system\dataset\images_resized')
newImgHeight = 256
newImgWidth = 256

# run this block of code to change size of the training images
"""for i in df_train.index:
#for i in range(0, 8):  # for test purposes we don't use the whole dataset
    img = cv2.imread(str(df_train['filename'][i]))
    cropped = img[int(df_train['ymin'][i]):int(df_train['ymax'][i]), int(df_train['xmin'][i]):int(df_train['xmax'][i])]
    resized = cv2.resize(cropped, (newImgWidth, newImgHeight), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(train_path_resized, str(df_train['filename'][i])[40:]), resized)
    df_train.at[i, 'filename'] = os.path.join(train_path_resized, str(df_train['filename'][i])[40:])
    df_train.at[i, 'width'] = newImgWidth
    df_train.at[i, 'height'] = newImgHeight"""

for i in df_train.index:
    df_train.at[i, 'filename'] = os.path.join(train_path_resized, str(df_train['filename'][i])[40:])
    df_train.at[i, 'width'] = newImgWidth
    df_train.at[i, 'height'] = newImgHeight

df_train = df_train.reset_index()

X = df_train['filename']
Y = df_train['class']

# creating training and validation dataset
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# parameters
number_epochs = 15
batch_size = 32
learning_rate = 0.002


class CarTruckDataset(Dataset):
    def __init__(self, paths, y, transform=None):
        self.paths = paths.values
        self.y = y.values
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = plt.imread(str(self.paths[idx])).copy()

        if self.transform:
            img = self.transform(img)

        return img, self.y[idx]


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


train_ds = CarTruckDataset(X_train, y_train, transform=transform)
valid_ds = CarTruckDataset(X_val, y_val, transform=transform)

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=batch_size)

# these two lines for test purposes
"""dataiter = iter(train_dl)
images, labels = next(dataiter)"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)


# CNN model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 28 * 28, 250)
        self.fc2 = nn.Linear(250, 95)
        self.fc3 = nn.Linear(95, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 32*28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_dl)

for epoch in range(number_epochs):
    for i, (images, labels) in enumerate(train_dl):

        # original shape of images: (75, 3, 256, 256)
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{number_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finished Training')

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(2)]
    n_class_samples = [0 for i in range(2)]
    for images, labels in valid_dl:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(len(labels)):
            label = labels[i]
            pred = predicted[i]
            if label == pred:
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    classes = ('car', 'truck')

    for i in range(2):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')

toSave = input("Do you want to save the model? (y/n)")
if toSave == "y":
    FILE = "model.pth"
    torch.save(model.state_dict(), FILE)
