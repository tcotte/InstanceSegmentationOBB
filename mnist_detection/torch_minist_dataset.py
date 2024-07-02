import json
import os
from typing import List

import torch
from imutils import paths
import cv2
import imutils
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from mnist_detection.utils import transform_xywh_to_3D_annotation, transform_xywh_to_3D_onehotencode_annotation


class MnistBoundingBoxes(Dataset):
    def __init__(self, folder_path, from_npy=False, transforms=None):
        # store the image and mask filepaths, and augmentation
        # transforms
        self.folder_path = folder_path
        self.transforms = transforms
        self.from_npy = from_npy

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(list(imutils.paths.list_images(self.folder_path)))

    def __getitem__(self, idx):

        img_name = list(imutils.paths.list_images(self.folder_path))[idx]
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.from_npy:
            ann_file = os.path.join(self.folder_path, img_name.replace(".jpg", ".npy"))
            annotations = np.load(ann_file)

        else:
            ann_file = os.path.join(self.folder_path, img_name.replace(".jpg", ".json"))

            with open(ann_file, encoding="utf8") as json_file:
                file_content = json_file.read()

            json_file.close()
            parsed_json = json.loads(file_content)
            objects = parsed_json["objects"]

            nb_classes = 10
            annotations = np.zeros((8, 8, 5 + nb_classes), dtype=np.float32)
            for obj in objects:
                bbox = obj["points"]["exterior"]
                bbox_xywh = [bbox[0][0] + 14, bbox[0][1] + 14, 28, 28]
                threeD_ann = transform_xywh_to_3D_onehotencode_annotation(xywh_box=bbox_xywh, cls_id=obj['classId'],
                                                                          grid_size=16, grid_rows=8, grid_cols=8,
                                                                          nb_classes=10)

                possibility_add = True
                for idx in np.flatnonzero(annotations[..., 0]):
                    if idx == np.flatnonzero(threeD_ann[..., 0]):
                        possibility_add = False

                if possibility_add:
                    annotations += threeD_ann

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
    dataset = MnistBoundingBoxes(
        folder_path=r"C:\Users\tristan_cotte\PycharmProjects\InstanceSegmentationOBB\mnist_detection\dataset\train",
        from_npy=False, transforms=None)
    print(dataset[0][0].size(), dataset[0][1])

    dataset_npy = MnistBoundingBoxes(
        folder_path=r"C:\Users\tristan_cotte\PycharmProjects\InstanceSegmentationOBB\mnist_detection\dataset\train",
        from_npy=True, transforms=None)

    print(dataset_npy[0][0].size(), dataset_npy[0][1])

    print(dataset[0][1] == dataset_npy[0][1])

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
