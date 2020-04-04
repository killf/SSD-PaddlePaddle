import paddle.fluid as fluid
import numpy as np


def focal_loss(pred, gt, gamma=4, soft_label=False, numeric_stable_mode=True, return_softmax=False):
    if not soft_label:
        gt = fluid.layers.one_hot(gt, depth=pred.shape[1])

    if numeric_stable_mode:
        pred = fluid.layers.clip(pred, min=1e-7, max=1. - 1e-7)

    pred = fluid.layers.softmax(pred)

    loss = -1 * gt * fluid.layers.log(pred) * fluid.layers.pow(1. - pred, gamma)
    loss = fluid.layers.reduce_sum(loss, dim=1, keep_dim=True)

    if return_softmax:
        return loss, pred
    else:
        return loss
