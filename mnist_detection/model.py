import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from mnist_detection.dataset_creator import make_data, show_predict


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)

        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(32)

        self.conv_prob = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.conv_boxes = nn.Conv2d(32, 4, kernel_size=3, padding=1)
        self.conv_cls = nn.Conv2d(32, 10, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.bn1(x)

        x = F.relu(self.conv2(x))
        x = self.bn2(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.bn3(x)

        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.bn4(x)

        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = self.bn5(x)

        x_prob = torch.sigmoid(self.conv_prob(x))
        x_boxes = self.conv_boxes(x)
        x_cls = torch.sigmoid(self.conv_cls(x))

        gate = torch.where(x_prob > 0.5, torch.ones_like(x_prob), torch.zeros_like(x_prob))
        x_boxes = x_boxes * gate
        x_cls = x_cls * gate

        x = torch.cat([x_prob, x_boxes, x_cls], dim=1)

        # Permute the tensor to match the desired output shape [batch_size, 8, 8, 15]
        x = x.permute(0, 2, 3, 1)

        return x

if __name__ == '__main__':
    # Example usage
    model = MyModel()
    # x_input = torch.randn(1, 3, 128, 128)  # Batch size of 1, 3 channels, 128x128 image
    # output = model(x_input)
    # print(output.shape)



    X, y = make_data(size=1)
    plt.imshow(X.squeeze())
    plt.show()
    # print(torch.from_numpy(X).size())
    y_hat = model(torch.from_numpy(X).permute(0, 3, 1, 2))
    # plt.imshow(X[0])
    # plt.show()
    show_predict(X[0], y_hat.detach().numpy()[0])
    plt.show()
