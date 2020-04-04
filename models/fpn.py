import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import *


class FeaturePyramidNeck(fluid.dygraph.Layer):
    def __init__(self, num_fpn_filters):
        super(FeaturePyramidNeck, self).__init__()
        self.upSample = lambda x: fluid.layers.image_resize(x, scale=2)

        # no Relu for downsample layer
        self.downSample1 = Conv2D(num_fpn_filters, num_fpn_filters, 3, 2, padding=1)
        self.downSample2 = Conv2D(num_fpn_filters, num_fpn_filters, 3, 2, padding=1)

        self.lateralCov1 = Conv2D(2048, num_fpn_filters, 1, 1, act="relu")
        self.lateralCov2 = Conv2D(1024, num_fpn_filters, 1, 1, act="relu")
        self.lateralCov3 = Conv2D(512, num_fpn_filters, 1, 1, act="relu")

        # predict layer for FPN
        self.predictP5 = Conv2D(num_fpn_filters, num_fpn_filters, 3, 1, padding=1)
        self.predictP4 = Conv2D(num_fpn_filters, num_fpn_filters, 3, 1, padding=1)
        self.predictP3 = Conv2D(num_fpn_filters, num_fpn_filters, 3, 1, padding=1)

    def forward(self, c3, c4, c5):
        # lateral conv for c3 c4 c5
        p5 = self.lateralCov1(c5)
        p4 = self._crop_and_add(self.upSample(p5), self.lateralCov2(c4))
        p3 = self._crop_and_add(self.upSample(p4), self.lateralCov3(c3))
        # print("p3: ", p3.shape)

        # smooth pred layer for p3, p4, p5
        p3 = self.predictP3(p3)
        p4 = self.predictP4(p4)
        p5 = self.predictP5(p5)

        # downsample conv to get p6, p7
        p6 = self.downSample1(p5)
        p7 = self.downSample2(p6)

        return [p3, p4, p5, p6, p7]

    def _crop_and_add(self, x1, x2):
        """
        for p4, c4; p3, c3 to concatenate with matched shape
        https://tf-unet.readthedocs.io/en/latest/_modules/tf_unet/layers.html
        """
        _, _, h1, w1 = x1.shape
        _, _, h2, w2 = x2.shape

        dh = (h1 - h2) // 2
        dw = (w1 - w2) // 2

        x1_crop = x1[:, :, dh:dh + h2, dw:dw + w2]
        return fluid.layers.sums([x1_crop, x2])
