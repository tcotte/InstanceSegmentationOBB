import os

import torch
import torch.optim as optim
from torch import nn

from mnist_detection.model import MyModel
from mnist_detection.torch_minist_dataset import MnistBoundingBoxes

# Enable CUDA launch blocking
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

batch_size = 32

train_dataset = MnistBoundingBoxes(
    folder_path=r"C:\Users\tristan_cotte\PycharmProjects\InstanceSegmentationOBB\mnist_detection\dataset\train",
    transforms=None)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=2)

test_dataset = MnistBoundingBoxes(
    folder_path=r"C:\Users\tristan_cotte\PycharmProjects\InstanceSegmentationOBB\mnist_detection\dataset\val",
    transforms=None)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=2)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MyModel()
model = model.to(device)

loss_bounding_boxes = nn.MSELoss(reduction="sum")
loss_prediction = nn.BCELoss(reduction="sum")
loss_cls = nn.CrossEntropyLoss(reduction="sum")

# criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

nb_epochs = 10

if __name__ == "__main__":
    for epoch in range(nb_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        running_proba_loss = 0.0
        running_bbox_loss = 0.0
        running_category_loss = 0.0

        model.train()
        for i, data in enumerate(train_dataloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            # outputs = torch.reshape(outputs, (batch_size, 8, 8, 15))
            # outputs = torch.permute(outputs, (0, 2, 3, 1))

            output_prob = outputs[..., 0]  # Shape: (batch_size, grid_size, grid_size)
            labels_prob = labels[..., 0]
            # sig_output_prob = torch.sigmoid(output_prob)

            loss_pred = loss_prediction(output_prob, labels_prob)

            output_bboxes = outputs[..., 1:5]
            label_bboxes = labels[..., 1:5]
            loss_regression_box = loss_bounding_boxes(output_bboxes, label_bboxes)

            output_category = outputs[..., 6:]
            label_category = labels[..., 6:]
            loss_category = loss_cls(output_category, label_category)

            loss = loss_pred + loss_regression_box + loss_category
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_proba_loss += loss_pred.item()
            running_bbox_loss += loss_regression_box.item()
            running_category_loss += loss_category.item()
            if i % 10 == 0 and i > 1:  # print every 10 mini-batches
                print(
                    f'[{epoch + 1} epoch, batch: {i + 1:5d}] total loss: {running_loss / 10:.3f} / proba loss: {running_proba_loss / 10:.3f} / bbox loss: {running_bbox_loss / 10:.3f} / category loss: {running_category_loss / 10:.3f}')
                running_loss = 0.0
                running_proba_loss = 0.0
                running_bbox_loss = 0.0
                running_category_loss = 0.0

    print('Finished Training')

    torch.save(model.state_dict(), "models/mnist_detection.pt")
