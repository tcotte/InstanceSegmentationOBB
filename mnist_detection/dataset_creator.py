# preapre handwritten digits
import base64
import io
import json
import os
import zlib

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

from mnist_detection.utils import transform_npy_to_json

(X_num, y_num), _ = tf.keras.datasets.mnist.load_data()
X_num = np.expand_dims(X_num, axis=-1).astype(np.float32) / 255.0

grid_size = 16  # image_size / mask_size


def mask_2_base64(mask):
    """
    https://github.com/supervisely/docs/blob/master/data-organization/Annotation-JSON-format/04_Supervisely_Format_objects.md#bitmap
    :param mask:
    :return:
    """
    img_pil = Image.fromarray(np.array(mask, dtype=np.uint8))
    img_pil.putpalette([0,0,0,255,255,255])
    bytes_io = io.BytesIO()
    img_pil.save(bytes_io, format='PNG', transparency=0, optimize=0)
    bytes = bytes_io.getvalue()
    return base64.b64encode(zlib.compress(bytes)).decode('utf-8')


def make_numbers(X, y, mask):
    positions_raw_col = []
    for i in range(3):
        # pickup random index
        idx = np.random.randint(len(X_num))

        # make digit colorful
        number = X_num[idx] @ (np.random.rand(1, 3) + 0.1)
        number[number > 0.1] = np.clip(number[number > 0.1], 0.5, 0.8)

        # Image.fromarray(np.ceil(np.array(number))[:, :, 0].astype(np.uint8))

        # class of digit
        kls = y_num[idx]

        # random position for digit
        px, py = np.random.randint(0, 100), np.random.randint(0, 100)

        # digit belong which mask position
        mx, my = (px + 14) // grid_size, (py + 14) // grid_size

        if [mx, my] not in positions_raw_col:
            channels = y[my][mx]

            # prevent duplicated problem
            if channels[0] > 0:
                continue

            channels[0] = 1.0
            channels[1] = px - (mx * grid_size)  # x1
            channels[2] = py - (my * grid_size)  # y1
            channels[3] = 28.0  # x2, in this demo image only 28 px as width
            channels[4] = 28.0  # y2, in this demo image only 28 px as height
            channels[5 + kls] = 1.0

            # put digit in X
            X[py:py + 28, px:px + 28] += number

            mask["objects"].append({
                "classId": int(kls),
                "classTitle": str(kls),
                "geometryType": "bitmap",
                "bitmap": {
                    "data": mask_2_base64(np.ceil(np.array(number))[:, :, 0].astype(np.uint8)),
                    "origin": [px, py]
                }
            })

            positions_raw_col.extend([[mx, my], [mx-1, my], [mx+1, my], [mx, my-1], [mx, my+1], [my-1, mx-1], [my+1, mx+1],
                                     [mx-1, my+1], [mx+1, mx-1]])


def make_data(size=64):
    X = np.zeros((size, 128, 128, 3), dtype=np.float32)
    y = np.zeros((size, 8, 8, 15), dtype=np.float32)
    masks = np.zeros((size, 128, 128), dtype=np.uint8)
    for i in range(size):
        make_numbers(X[i], y[i], masks[i])

    X = np.clip(X, 0.0, 1.0)
    return X, y


def get_color_by_probability(p):
    if p < 0.3:
        return (1., 0., 0.)
    if p < 0.7:
        return (1., 1., 0.)
    return (0., 1., 0.)


def show_predict(X, y, threshold=0.1):
    X = X.copy()
    for mx in range(8):
        for my in range(8):
            channels = y[my][mx]
            prob, x1, y1, x2, y2 = channels[:5]

            if prob > threshold:
                print(channels)
            # if prob < 0.1 we won't show anything
            if prob < threshold:
                continue

            color = get_color_by_probability(prob)
            # bounding box
            px, py = (mx * grid_size) + x1, (my * grid_size) + y1
            cv2.rectangle(X, (int(px), int(py)), (int(px + x2), int(py + y2)), color, 1)

            # label
            cv2.rectangle(X, (int(px), int(py - 10)), (int(px + 12), int(py)), color, -1)
            kls = np.argmax(channels[5:])
            cv2.putText(X, f'{kls}', (int(px + 2), int(py - 2)), cv2.FONT_HERSHEY_PLAIN, 0.7, (0.0, 0.0, 0.0))

    plt.imshow(X)


def create_dataset(path_dataset_folder, nb_items):
    if not os.path.isdir(path_dataset_folder):
        os.makedirs(path_dataset_folder)

    for i in range(nb_items):
        X = np.zeros((128, 128, 3), dtype=np.float32)
        y = np.zeros((8, 8, 15), dtype=np.float32)
        mask = {"objects": []}
        make_numbers(X, y, mask)
        cv2.imwrite(os.path.join(path_dataset_folder, f'img_{i}.jpg'), X*255)
        np.save(os.path.join(path_dataset_folder, f'img_{i}.npy'), y)

        with open(os.path.join(path_dataset_folder, f'mask_{i}.json'), 'w', encoding='utf8') as json_file:
            json.dump(mask, json_file, ensure_ascii=False)

        # cv2.imwrite(os.path.join(path_dataset_folder, f'mask_{i}.png'), mask * 255)


# def transform_anchor_annotation2bbox(anchor_annotation):



# test
if __name__ == "__main__":
    # size = 1
    # X, y = make_data(size=size)
    # for i in range(size):
    #     show_predict(X[i], y[i])
    #     print(y[i].shape)
    #     print(y[i])
    #     plt.show()


    create_dataset("dataset/train", nb_items=400*32)
    create_dataset("dataset/val", nb_items=10*32)

    list_path_annotations = ["dataset/train", "dataset/val"]

    for path_annotations in list_path_annotations:
        for i in os.listdir(path_annotations):
            if i.endswith(".npy"):
                json_data = transform_npy_to_json(os.path.join(path_annotations, i))
                outfile = open(os.path.join(path_annotations, i[:-4] + ".json"), "w")
                json.dump(json_data, outfile)
                outfile.close()

    # results = np.load(r"C:\Users\tristan_cotte\PycharmProjects\InstanceSegmentationOBB\mnist_detection\img_35.npy")
    # print(results, results.shape)
