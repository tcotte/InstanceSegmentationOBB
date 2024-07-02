import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

from mnist_detection.dataset_creator import show_predict
from mnist_detection.model import MyModel
from mnist_detection.torch_minist_dataset import MnistBoundingBoxes

from mnist_detection.dataset_creator import show_predict

test_dataset = MnistBoundingBoxes(folder_path=r"C:\Users\tristan_cotte\PycharmProjects\InstanceSegmentationOBB\mnist_detection\dataset\val",
                                  transforms=None)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                              shuffle=False, num_workers=2)

if __name__ == "__main__":
    path_model = r"C:\Users\tristan_cotte\PycharmProjects\InstanceSegmentationOBB\mnist_detection\models\mnist_detection.pt"
    model = MyModel()
    model.load_state_dict(torch.load(path_model))
    model.eval()

    # img = cv2.imread(r"C:\Users\tristan_cotte\PycharmProjects\InstanceSegmentationOBB\mnist_detection\dataset\val\img_0.jpg")
    # rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for idx, data in enumerate(test_dataloader):
        # imgs, target = data
        if idx > 30:
            break

        img, label = data
        cv2_img = np.squeeze(img.numpy()).transpose(1, 2, 0)
        model(img)

        predicted_annotations = np.squeeze(model(img).detach().cpu().numpy())
        print(np.where(predicted_annotations[:, 0 ] > 0.2) )

        show_predict(X=cv2_img, y=predicted_annotations, threshold=0.9)
        plt.show()