import os
from typing import List

import torch
from imutils import paths
import cv2
import imutils
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class MnistBoundingBoxes(Dataset):
    def __init__(self, folder_path, transforms):
        # store the image and mask filepaths, and augmentation
        # transforms
        self.folder_path = folder_path
        self.transforms = transforms

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(list(imutils.paths.list_images(self.folder_path)))

    def __getitem__(self, idx):

        img_name = list(imutils.paths.list_images(self.folder_path))[idx]
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_file = os.path.join(self.folder_path, img_name.replace(".jpg", ".npy"))
        annotations = np.load(ann_file)

        # check to see if we are applying any transformations
        if self.transforms is not None:
            # apply the transformations to both image and its mask
            transformed = self.transforms(image=image, bounding_boxes=annotations)
            image = transformed['image']
            annotations = transformed['bounding_boxes']

        else:
            transform = transforms.ToTensor()
            # Convert the image to PyTorch tensor
            image = transform(image)
            annotations = torch.tensor(annotations)



        # return a tuple of the image and its mask
        return image, annotations

if __name__ == "__main__":
    dataset = MnistBoundingBoxes(folder_path=r"C:\Users\tristan_cotte\PycharmProjects\InstanceSegmentationOBB\mnist_detection\dataset\val", transforms=None)
    print(dataset[0][0].size(), dataset[0][1])

    # def collate_fn(self, batch):
    #     """
    #     Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
    #
    #     This describes how to combine these tensors of different sizes. We use lists.
    #
    #     Note: this need not be defined in this Class, can be standalone.
    #
    #     :param batch: an iterable of N sets from __getitem__()
    #     :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
    #     """
    #
    #     images = []
    #     labels = []
    #     masks = []
    #     obb_point_based = []
    #     obb_teta_based = []
    #     # boxes = []
    #     # labels = list()
    #     # difficulties = list()
    #
    #     for b in batch:
    #         images.append(b[0])
    #         labels.append(b[1]["labels"])
    #         obb_point_based.append(b[1]["obb_point_based"])
    #         obb_teta_based.append(b[1]["obb_teta_based"])
    #
    #     images = torch.stack(images, dim=0)
    #
    #     return images, labels, obb_point_based