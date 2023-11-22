import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


def image_classifier(image):

    file = "model.pth"
    device = torch.device('cpu')

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

    model = ConvNet()

    img_to_show = image.copy()

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    image = transform(image)

    model.load_state_dict(torch.load(file, map_location=device))

    with torch.no_grad():
        model.eval()
        result = model(image)
        _, predicted = torch.max(result, 1)

    classes = ('car', 'truck')

    # for testing
    # img_to_show = cv2.putText(img_to_show, classes[predicted], (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    # cv2.imshow("image", img_to_show)

    return classes[predicted] == 'truck'


if __name__ == "__main__":
    img = cv2.imread("own_dataset/car/1.jpg")
    print(image_classifier(img))
