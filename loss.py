import paddle.fluid as fluid
from data.anchor import Anchor
from config import cfg
import numpy as np
import cv2


class YolactLoss(object):
    def __init__(self, loss_weight_cls=1,
                 loss_weight_box=1.5,
                 neg_pos_ratio=3,
                 max_masks_for_train=100):
        self._loss_weight_cls = loss_weight_cls
        self._loss_weight_box = loss_weight_box
        self._neg_pos_ratio = neg_pos_ratio
        self._max_masks_for_train = max_masks_for_train

        self._output_size = cfg.image_size
        self._anchor_instance = Anchor(img_size=cfg.image_size,
                                       feature_map_size=[69, 35, 18, 9, 5],
                                       aspect_ratio=[1, 0.5, 2],
                                       scale=[24, 48, 96, 192, 384])
        self._match_threshold = 0.5
        self._unmatched_threshold = 0.4
        self._proto_output_size = 138
        self._num_max_fix_padding = 100

    def prepare(self, label):
        targets, masks, num_crowds = label

        boxes = fluid.dygraph.to_variable(targets[:, :4].astype(np.float32))
        classes = fluid.dygraph.to_variable(targets[:, -1].astype(np.int64))

        boxes_norm = boxes * (1.0 * self._proto_output_size / self._output_size)
        cls_targets, box_targets, max_id_for_anchors, match_positiveness = self._anchor_instance.matching(
            self._match_threshold, self._unmatched_threshold, boxes, classes)

        return {
            'cls_targets': cls_targets,
            'box_targets': box_targets,
            'bbox': boxes,
            'bbox_for_norm': boxes_norm,
            'positiveness': match_positiveness,
            'classes': classes,
            'mask_target': masks,
            'max_id_for_anchors': max_id_for_anchors
        }

    def __call__(self, batch_pred, batch_label, num_classes):
        batch_size = len(batch_label[0])
        loc_loss_ls, conf_loss_ls = [], []
        for idx in range(batch_size):
            pred = {k: v[idx] for k, v in batch_pred.items()}
            label = tuple([v[idx] for v in batch_label])

            # all prediction component
            # pred_cls, pred_offset = pred
            pred_cls = pred['pred_cls']
            pred_offset = pred['pred_offset']

            # all label component
            label = self.prepare(label)
            cls_targets = label['cls_targets']
            box_targets = label['box_targets']
            positiveness = label['positiveness']

            loc_loss = self._loss_location(pred_offset, box_targets, positiveness)
            conf_loss = self._loss_class(pred_cls, cls_targets, num_classes, positiveness)

            loc_loss_ls.append(loc_loss)
            conf_loss_ls.append(conf_loss)

        loc_loss = fluid.layers.sums(loc_loss_ls) / batch_size
        conf_loss = fluid.layers.sums(conf_loss_ls) / batch_size

        total_loss = self._loss_weight_box * loc_loss + self._loss_weight_cls * conf_loss
        return loc_loss, conf_loss, total_loss

    def _loss_location(self, pred_offset, gt_offset, positiveness):
        # get postive indices
        pos_indices = fluid.layers.where(positiveness == 1)

        # pred_offset = fluid.layers.squeeze(pred_offset, [0])
        pred_offset = fluid.layers.gather_nd(pred_offset, pos_indices)
        gt_offset = fluid.layers.gather_nd(gt_offset, pos_indices)

        # calculate the smoothL1(positive_pred, positive_gt) and return
        loss_loc = fluid.layers.smooth_l1(gt_offset, pred_offset)
        loss_loc = fluid.layers.reduce_mean(loss_loc)

        return loss_loc

    def _loss_class(self, pred_cls, gt_cls, num_cls, positiveness):
        # reshape gt_cls from [batch, num_anchor] => [batch * num_anchor, 1]
        gt_cls = fluid.layers.unsqueeze(gt_cls, [1])

        # reshape positiveness to [batch*num_anchor, 1]
        pos_indices = fluid.layers.where(positiveness == 1)
        neg_indices = fluid.layers.where(positiveness == 0)

        # calculate the needed amount of  negative sample
        num_pos = pos_indices.shape[0]
        num_neg_needed = num_pos * self._neg_pos_ratio

        # gather pos data, neg data separately
        pos_pred_cls = fluid.layers.gather(pred_cls, pos_indices)
        pos_gt = fluid.layers.gather(gt_cls, pos_indices)

        neg_pred_cls = fluid.layers.gather(pred_cls, neg_indices)
        neg_gt = fluid.layers.gather(gt_cls, neg_indices)

        # # apply softmax on the pred_cls
        neg_pred_cls_softmax = fluid.layers.softmax(neg_pred_cls)
        _, neg_minus_log_class0_sort = fluid.layers.argsort(neg_pred_cls_softmax[:, 0], descending=False)

        # take the first num_neg_needed idx in sort result and handle the situation if there are not enough neg
        neg_indices_for_loss = neg_minus_log_class0_sort[:num_neg_needed]

        # combine the indices of pos and neg sample, create the label for them
        neg_pred_cls_for_loss = fluid.layers.gather(neg_pred_cls, neg_indices_for_loss)
        neg_gt_for_loss = fluid.layers.gather(neg_gt, neg_indices_for_loss)

        # calculate Cross entropy loss and return
        # concat positive and negtive data
        target_logits = fluid.layers.concat([pos_pred_cls, neg_pred_cls_for_loss], axis=0)
        target_labels = fluid.layers.cast(fluid.layers.concat([pos_gt, neg_gt_for_loss], axis=0), "int64")

        loss_conf = fluid.layers.softmax_with_cross_entropy(label=target_labels, logits=target_logits)
        loss_conf = fluid.layers.reduce_mean(loss_conf)

        return loss_conf
