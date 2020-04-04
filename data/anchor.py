from itertools import product
from math import sqrt
import paddle.fluid as fluid
import numpy as np


def where(cond, x, y):
    return fluid.dygraph.to_variable(np.where(cond.numpy(), x.numpy(), y.numpy()))


# Can generate one instance only when creating the model
class Anchor(object):

    def __init__(self, img_size, feature_map_size, aspect_ratio, scale):
        """
        :param img_size:
        :param feature_map_size:
        :param aspect_ratio:
        :param scale:
        """
        self.num_anchors, self.anchors = self._generate_anchors(img_size, feature_map_size, aspect_ratio, scale)

    def _generate_anchors(self, img_size, feature_map_size, aspect_ratio, scale):
        """
        :param img_size:
        :param feature_map_size:
        :param aspect_ratio:
        :param scale:
        :return:
        """
        prior_boxes = []
        num_anchors = 0
        for idx, f_size in enumerate(feature_map_size):
            # print("Create priors for f_size:%s", f_size)
            count_anchor = 0
            for j, i in product(range(f_size), range(f_size)):
                f_k = img_size / (f_size + 1)
                x = f_k * (i + 1)
                y = f_k * (j + 1)
                for ars in aspect_ratio:
                    a = sqrt(ars)
                    w = scale[idx] * a
                    h = scale[idx] / a
                    # directly use point form here => [xmin, ymin, xmax, ymax]
                    ymin = max(0, y - (h / 2))
                    xmin = max(0, x - (w / 2))
                    ymax = min(img_size, y + (h / 2))
                    xmax = min(img_size, x + (w / 2))
                    prior_boxes += [xmin, ymin, xmax, ymax]
                    count_anchor += 1
            num_anchors += count_anchor

        prior_boxes = np.array(prior_boxes, dtype=np.float32)
        output = fluid.layers.reshape(fluid.dygraph.to_variable(prior_boxes), [num_anchors, 4])
        return num_anchors, output

    def _pairwise_intersection(self, anchor, gt_bbox):
        """
        ref: https://github.com/tensorflow/models/blob/831281cedfc8a4a0ad7c0c37173963fafb99da37/official/vision/detection/utils/object_detection/box_list_ops.py
        :param gt_bbox: [num_obj, 4]
        :return:
        """
        # unstack to (x1, y1, x2, y2)
        x1_anchor, y1_anchor, x2_anchor, y2_anchor = fluid.layers.unstack(anchor, axis=-1)
        x1_gt, y1_gt, x2_gt, y2_gt = fluid.layers.unstack(gt_bbox, axis=-1)

        # calculate intersection
        all_pairs_max_x1 = fluid.layers.elementwise_max(x1_anchor, x1_gt)
        all_pairs_min_x2 = fluid.layers.elementwise_min(x2_anchor, x2_gt)
        all_pairs_max_y1 = fluid.layers.elementwise_max(y1_anchor, y1_gt)
        all_pairs_min_y2 = fluid.layers.elementwise_min(y2_anchor, y2_gt)
        intersect_heights = fluid.layers.elementwise_max(fluid.layers.zeros_like(all_pairs_min_y2),
                                                         all_pairs_min_y2 - all_pairs_max_y1)
        intersect_widths = fluid.layers.elementwise_max(fluid.layers.zeros_like(all_pairs_min_x2),
                                                        all_pairs_min_x2 - all_pairs_max_x1)

        return intersect_heights * intersect_widths

    def _pairwise_iou(self, gt_bbox):
        """ˇ
         ref: https://github.com/tensorflow/models/blob/831281cedfc8a4a0ad7c0c37173963fafb99da37/official/vision/detection/utils/object_detection/box_list_ops.py
        :param gt_bbox: [num_obj, 4]
        :return:
        """
        # get the num of gt_bbox and anchor, and reshape to the same shape.
        num_bbox = gt_bbox.shape[0]  # shape is [num_bbox, 4]
        num_anchor = self.anchors.shape[0]  # shape is [num_anchor, 4]

        anchor2 = fluid.layers.reshape(self.anchors, [num_anchor, 1, 4])
        anchor2 = fluid.layers.expand(anchor2, [1, num_bbox, 1])

        gt_bbox2 = fluid.layers.reshape(gt_bbox, [1, num_bbox, 4])
        gt_bbox2 = fluid.layers.expand(gt_bbox2, [num_anchor, 1, 1])

        # A ∩ B / A ∪ B = A ∩ B / (areaA + areaB - A ∩ B)
        # calculate A ∩ B (pairwise)
        pairwise_inter = self._pairwise_intersection(anchor=anchor2, gt_bbox=gt_bbox2)

        # calculate areaA, areaB
        x1_anchor, y1_anchor, x2_anchor, y2_anchor = fluid.layers.unstack(anchor2, axis=-1)
        x1_gt, y1_gt, x2_gt, y2_gt = fluid.layers.unstack(gt_bbox2, axis=-1)

        area_anchor = (x2_anchor - x1_anchor) * (y2_anchor - y1_anchor)
        area_gt = (x2_gt - x1_gt) * (y2_gt - y1_gt)

        # calculate A ∪ B = areaA + areaB - A ∩ B
        pairwise_union = area_anchor + area_gt - pairwise_inter

        # IOU(Jaccard overlap) = intersection / union
        iou = pairwise_inter / pairwise_union
        return iou

    def get_anchors(self):
        return self.anchors

    def matching(self, threshold_pos, threshold_neg, gt_bbox, gt_labels):
        """
        :param threshold_neg:
        :param threshold_pos:
        :param gt_bbox:
        :param gt_labels:
        :return:
        """
        # calculate iou
        pairwise_iou = self._pairwise_iou(gt_bbox=gt_bbox)

        # assign the max overlap gt index for each anchor
        max_iou_for_anchors = fluid.layers.reduce_max(pairwise_iou, dim=-1)
        max_id_for_anchors = fluid.layers.argmax(pairwise_iou, axis=-1)

        # decide the anchors to be positive or negative based on the IoU and given threshold
        max_iou_for_anchors_np = max_iou_for_anchors.numpy()
        if max_iou_for_anchors_np.max() < threshold_pos:
            threshold_pos = max_iou_for_anchors_np.max()
            threshold_neg = min(threshold_neg, threshold_pos * 0.9)

        pos_cond = max_iou_for_anchors_np >= threshold_pos
        neg_cond = max_iou_for_anchors_np <= threshold_neg
        match_positiveness_np = np.copy(max_iou_for_anchors_np)
        match_positiveness_np[pos_cond] = 1
        match_positiveness_np[neg_cond] = 0
        match_positiveness_np[np.logical_and(np.logical_not(pos_cond), np.logical_not(neg_cond))] = -1
        match_positiveness = fluid.dygraph.to_variable(match_positiveness_np.astype(np.int))

        assert fluid.layers.where(match_positiveness == 1).shape[0] > 0

        """
        create class target, map idx to label[idx]
        element-wise multiplication of label[idx] and positiveness:
        1. positive sample will have correct label
        2. negative sample will have 0 * label[idx] = 0
        3. neural sample will have -1 * label[idx] = -1 * label[idx] 
        it can be useful to distinguish positive sample during loss calculation  
        """
        match_labels = fluid.layers.gather(gt_labels, max_id_for_anchors)
        target_cls = fluid.layers.elementwise_mul(match_labels, match_positiveness)

        # create loc target
        map_loc = fluid.layers.gather(gt_bbox, max_id_for_anchors)

        # convert to center form
        w = self.anchors[:, 2] - self.anchors[:, 0]
        h = self.anchors[:, 3] - self.anchors[:, 1]
        center_anchors = fluid.layers.stack([self.anchors[:, 0] + (w / 2), self.anchors[:, 1] + (h / 2), w, h], axis=-1)

        w = map_loc[:, 2] - map_loc[:, 0]
        h = map_loc[:, 3] - map_loc[:, 1]
        center_gt = fluid.layers.stack([map_loc[:, 0] + (w / 2), map_loc[:, 1] + (h / 2), w, h], axis=-1)

        # calculate offset
        g_hat_cx = (center_gt[:, 0] - center_anchors[:, 0]) / center_anchors[:, 2]
        g_hat_cy = (center_gt[:, 1] - center_anchors[:, 1]) / center_anchors[:, 3]
        g_hat_w = fluid.layers.log(center_anchors[:, 2] / center_gt[:, 2])
        g_hat_h = fluid.layers.log(center_anchors[:, 3] / center_gt[:, 3])
        target_loc = fluid.layers.stack([g_hat_cx, g_hat_cy, g_hat_w, g_hat_h], axis=-1)

        return target_cls, target_loc, max_id_for_anchors, match_positiveness
