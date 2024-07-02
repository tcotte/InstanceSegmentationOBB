import json
import os

import numpy as np

def transform_xywh_to_3D_annotation(xywh_box, grid_size=16, grid_rows=8, grid_cols=8):
    # Initialize the annotation array
    annotation_array = np.zeros((grid_rows, grid_cols, 5), dtype=np.float32)

    x, y, width, height = xywh_box

    # Calculate the grid cell indices
    mx = int(x // grid_size)
    my = int(y // grid_size)

    # Calculate relative coordinates within the grid cell
    relative_x = x - mx * grid_size
    relative_y = y - my * grid_size

    # Store the annotation in the corresponding grid cell
    annotation_array[mx, my] = [1, relative_x, relative_y, width, height]
    return annotation_array

def transform_xywh_to_3D_onehotencode_annotation(xywh_box, cls_id, grid_size=16, grid_rows=8, grid_cols=8, nb_classes=10):
    # Initialize the annotation array
    annotation_array = np.zeros((grid_rows, grid_cols, 5 + nb_classes), dtype=np.float32)

    x, y, width, height = xywh_box

    # Calculate the grid cell indices
    mx = int(x // grid_size)
    my = int(y // grid_size)

    # Calculate relative coordinates within the grid cell
    relative_x = x - mx * grid_size
    relative_y = y - my * grid_size

    annotation_cls = [0] * nb_classes
    annotation_cls[cls_id] = 1

    # Store the annotation in the corresponding grid cell
    annotation_array[mx, my] = [1, relative_x, relative_y, width, height] + annotation_cls
    return annotation_array


def transform_3D_annotation_to_xywh(threeD_annotation, grid_size=16, grid_rows=8, grid_cols=8):
    list_xywh_bbox = []
    for x in range(threeD_annotation.shape[0]):
        for y in range(threeD_annotation.shape[1]):
            if threeD_annotation[x][y][0] == 1:
                cls = np.argmax(threeD_annotation[x][y][5:])
                xywh_box = [x*grid_size + threeD_annotation[x][y][1],y*grid_size + threeD_annotation[x][y][2], 28, 28, cls]
                list_xywh_bbox.append(xywh_box)

    return list_xywh_bbox


def transform_npy_to_json(npy_file):
    threeD_annotation = np.load(npy_file)
    xywh_bboxes = transform_3D_annotation_to_xywh(threeD_annotation)

    json_data = {}
    object_list = []

    for bbox in xywh_bboxes:
        object_data = {
            "geometryType": "rectangle",
            "classId": int(bbox[4]),
            "classTitle": str(bbox[4]),
            "points": {
                "exterior": [
                    [int(bbox[0] - bbox[2]/2), int(bbox[1] - bbox[3]/2)],
                    [int(bbox[0] + bbox[2]/2), int(bbox[1] + bbox[3]/2)]
                ]
            }
        }
        object_list.append(object_data)

    json_data["objects"] = object_list
    return json_data


if __name__ == "__main__":
    # np_arr = np.load(r"C:\Users\tristan_cotte\PycharmProjects\InstanceSegmentationOBB\mnist_detection\dataset\train\img_0.npy")
    # print(np_arr)
    # bboxes = transform_3D_annotation_to_xywh(np_arr)
    # print(bboxes)
    #
    # print(transform_xywh_to_3D_annotation(bboxes[0]))
    #
    # print(transform_3D_annotation_to_xywh(transform_xywh_to_3D_annotation(bboxes[0])))

    path_annotations = r"C:\Users\tristan_cotte\PycharmProjects\InstanceSegmentationOBB\mnist_detection\dataset\val"
    for i in os.listdir(path_annotations):
        if i.endswith(".npy"):
            json_data = transform_npy_to_json(os.path.join(path_annotations, i))
            outfile = open(os.path.join(path_annotations, i[:-4] + ".json"), "w")
            json.dump(json_data, outfile)
            outfile.close()


