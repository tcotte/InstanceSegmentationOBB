# preapre handwritten digits
import os

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

(X_num, y_num), _ = tf.keras.datasets.mnist.load_data()
X_num = np.expand_dims(X_num, axis=-1).astype(np.float32) / 255.0

grid_size = 16  # image_size / mask_size


def make_numbers(X, y):
    for _ in range(3):
        # pickup random index
        idx = np.random.randint(len(X_num))

        # make digit colorful
        number = X_num[idx] @ (np.random.rand(1, 3) + 0.1)
        number[number > 0.1] = np.clip(number[number > 0.1], 0.5, 0.8)
        # class of digit
        kls = y_num[idx]

        # random position for digit
        px, py = np.random.randint(0, 100), np.random.randint(0, 100)

        # digit belong which mask position
        mx, my = (px + 14) // grid_size, (py + 14) // grid_size
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


def make_data(size=64):
    X = np.zeros((size, 128, 128, 3), dtype=np.float32)
    y = np.zeros((size, 8, 8, 15), dtype=np.float32)
    for i in range(size):
        make_numbers(X[i], y[i])

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
        make_numbers(X, y)
        cv2.imwrite(os.path.join(path_dataset_folder, f'img_{i}.jpg'), X*255)
        np.save(os.path.join(path_dataset_folder, f'img_{i}.npy'), y)


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

    # results = np.load(r"C:\Users\tristan_cotte\PycharmProjects\InstanceSegmentationOBB\mnist_detection\img_35.npy")
    # print(results, results.shape)
