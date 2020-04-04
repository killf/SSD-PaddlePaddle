import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import *


class PredictionModule(fluid.dygraph.Layer):

    def __init__(self, in_channels, out_channels, num_anchors, num_class):
        super(PredictionModule, self).__init__()
        self.num_anchors = num_anchors
        self.num_class = num_class

        self.Conv = Conv2D(in_channels, out_channels, 3, 1, padding=1, act="relu")

        self.classConv = Conv2D(in_channels, self.num_class * self.num_anchors, 3, 1, padding=1)

        self.boxConv = Conv2D(in_channels, 4 * self.num_anchors, 3, 1, padding=1)

    def forward(self, p):
        p = self.Conv(p)

        pred_class = self.classConv(p)
        pred_box = self.boxConv(p)

        pred_class = fluid.layers.reshape(pred_class, [pred_class.shape[0], self.num_class, 3, pred_box.shape[2], pred_box.shape[3]])
        pred_box = fluid.layers.reshape(pred_box, [pred_box.shape[0], 4, 3, pred_box.shape[2], pred_box.shape[3]])

        pred_class = fluid.layers.transpose(pred_class, [0, 3, 4, 2, 1])
        pred_box = fluid.layers.transpose(pred_box, [0, 3, 4, 2, 1])

        # reshape the prediction head result for following loss calculation
        pred_class = fluid.layers.reshape(pred_class, [pred_class.shape[0], -1, self.num_class])
        pred_box = fluid.layers.reshape(pred_box, [pred_box.shape[0], -1, 4])

        return pred_class, pred_box
