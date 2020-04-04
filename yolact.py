from models.resnet import ResNet
from models.fpn import FeaturePyramidNeck
from models.head import PredictionModule
from utils.create_prior import make_priors

import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import *


class Yolact(fluid.dygraph.Layer):
    def __init__(self, input_size, fpn_channels, feature_map_size, num_class, aspect_ratio,
                 scales):
        super(Yolact, self).__init__()

        # extract certain feature maps for FPN
        self.backbone = ResNet()
        self.fpn = FeaturePyramidNeck(fpn_channels)

        self.num_anchor, self.priors = make_priors(input_size, feature_map_size, aspect_ratio, scales)
        print("prior shape:", self.priors.shape)
        print("num anchor per feature map: ", self.num_anchor)

        # shared prediction head
        self.predictionHead = PredictionModule(fpn_channels, 256, len(aspect_ratio), num_class)

    def set_bn(self, mode='train'):
        if mode == 'train':
            for layer in self.backbone.layers:
                if isinstance(layer, BatchNorm):
                    layer.trainable = False
        else:
            for layer in self.backbone.layers:
                if isinstance(layer, BatchNorm):
                    layer.trainable = True

    def forward(self, inputs):
        # backbone(ResNet + FPN)
        c3, c4, c5 = self.backbone(inputs)
        # print("c3: ", c3.shape)
        # print("c4: ", c4.shape)
        # print("c5: ", c5.shape)
        fpn_out = self.fpn(c3, c4, c5)

        # Prediction Head branch
        pred_cls = []
        pred_offset = []

        # all output from FPN use same prediction head
        for idx, f_map in enumerate(fpn_out):
            cls, offset = self.predictionHead(f_map)
            pred_cls.append(cls)
            pred_offset.append(offset)

        pred_cls = fluid.layers.concat(pred_cls, axis=1)
        pred_offset = fluid.layers.concat(pred_offset, axis=1)

        return {
            'pred_cls': pred_cls,
            'pred_offset': pred_offset
        }
