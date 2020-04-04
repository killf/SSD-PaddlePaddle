import paddle.fluid as fluid
import numpy as np
import cv2


def bboxes_intersection(bbox_ref, bboxes):
    """Compute relative intersection between a reference box and a
    collection of bounding boxes. Namely, compute the quotient between
    intersection area and box area.
    Args:
      bbox_ref: (N, 4) or (4,) Tensor with reference bounding box(es).
      bboxes: (N, 4) Tensor, collection of bounding boxes.
    Return:
      (N,) Tensor with relative intersection.
    """

    # Should be more efficient to first transpose.
    bboxes = np.transpose(bboxes)
    bbox_ref = np.transpose(bbox_ref)

    # Intersection bbox and volume.
    int_ymin = np.maximum(bboxes[0], bbox_ref[0])
    int_xmin = np.maximum(bboxes[1], bbox_ref[1])
    int_ymax = np.minimum(bboxes[2], bbox_ref[2])
    int_xmax = np.minimum(bboxes[3], bbox_ref[3])
    h = np.maximum(int_ymax - int_ymin, 0.)
    w = np.maximum(int_xmax - int_xmin, 0.)

    # Volumes.
    inter_vol = h * w
    bboxes_vol = (bboxes[2] - bboxes[0]) * (bboxes[3] - bboxes[1])

    return np.where(
        np.equal(bboxes_vol, 0.0),
        np.zeros_like(inter_vol), inter_vol / bboxes_vol)


def normalize_image(image,
                    offset=(0.485, 0.456, 0.406),
                    scale=(0.229, 0.224, 0.225)):
    """Normalizes the image to zero mean and unit variance.
     ref: https://github.com/tensorflow/models/blob/3462436c91897f885e3593f0955d24cbe805333d/official/vision/detection/utils/input_utils.py
    """
    image = np.array(image).astype(np.float32)

    # offset = np.array(offset)
    # offset = np.expand_dims(offset, axis=0)
    # offset = np.expand_dims(offset, axis=0)
    # image -= offset
    #
    # scale = np.array(scale)
    # scale = np.expand_dims(scale, axis=0)
    # scale = np.expand_dims(scale, axis=0)
    # image *= scale

    image = image / 127.5 - 1.
    return image


def area_of_box(x):
    pass


def map_to_center_form(x):
    w = x[2] - x[0]
    h = x[3] - x[1]
    cx = x[0] + (w / 2)
    cy = x[1] + (h / 2)
    return np.stack([cx, cy, w, h])


def map_to_point_form(x):
    xmin = x[0] - (x[2] / 2)
    ymin = x[1] - (x[3] / 2)
    xmax = x[0] + (x[2] / 2)
    ymax = x[1] + (x[3] / 2)
    return np.stack([ymin, xmin, ymax, xmax])


# encode the gt and anchors to offset
def map_to_offset(x):
    g_hat_cx = (x[0, 0] - x[0, 1]) / x[2, 1]
    g_hat_cy = (x[1, 0] - x[1, 1]) / x[3, 1]
    g_hat_w = np.log(x[2, 0] / x[2, 1])
    g_hat_h = np.log(x[3, 0] / x[3, 1])
    return np.stack([g_hat_cx, g_hat_cy, g_hat_w, g_hat_h])


# decode the offset back to center form bounding box when evaluation and prediction
def map_to_bbox(x):
    pass


def single_pair_iou(pred, target):
    # IOU of single pair of bbox, for the purpose in NMS, pairwise IOU is implemented within "Anchor" class
    """
    :param pred:
    :param target:
    :return:
    """

    pass
