import os

import numpy as np
import paddle.fluid as fluid
from visualdl import LogWriter

from yolact import Yolact
from loss import YolactLoss
from data.coco import COCODetection, detection_collate
from data.parallel import DataLoader
from utils.augmentations import SSDAugmentation
from config import cfg

logdir = "./out/log"
logger = LogWriter(logdir, sync_cycle=3000)

# mark the components with 'train' label.
with logger.mode("train"):
    loc_loss_logger = logger.scalar("scalars/loc_loss")
    conf_loss_logger = logger.scalar("scalars/conf_loss")
    total_loss_logger = logger.scalar("scalars/total_loss")


def main():
    with fluid.dygraph.guard():
        # 1.加载数据集
        train_data = COCODetection(image_path=cfg.train_images,
                                   info_file=cfg.train_info,
                                   transform=SSDAugmentation(),
                                   batch_size=cfg.batch_size)
        data_loader = DataLoader(train_data, cfg.batch_size, shuffle=True, collate_fn=detection_collate)

        # 2.定义模型
        yolact = Yolact(input_size=cfg.image_size,
                        fpn_channels=256,
                        feature_map_size=[69, 35, 18, 9, 5],
                        num_class=cfg.num_class,
                        aspect_ratio=[1, 0.5, 2],
                        scales=[24, 48, 96, 192, 384])
        loss = YolactLoss()
        optimizer = fluid.optimizer.Adam(learning_rate=0.001,
                                         regularization=fluid.regularizer.L2Decay(cfg.weight_decay),
                                         parameter_list=yolact.parameters())

        if os.path.exists(cfg.model_file + ".pdparams") and os.path.exists(cfg.model_file + ".pdopt"):
            para_state_dict, opti_state_dict = fluid.dygraph.load_dygraph(cfg.model_file)
            yolact.set_dict(para_state_dict)
            optimizer.set_dict(opti_state_dict)

        # 3.开始训练
        total_step = 0
        for epoch in range(cfg.epochs):
            for step, data in enumerate(data_loader):
                total_step += 1

                image = np.stack(data[0])
                image = np.transpose(image, [0, 3, 1, 2])
                image = fluid.dygraph.to_variable(image)
                pred = yolact(image)

                loc_loss, conf_loss, total_loss = loss(pred, data[1], cfg.num_class)

                total_loss.backward()
                optimizer.minimize(total_loss)
                yolact.clear_gradients()

                loc_loss_logger.add_record(total_step, loc_loss.numpy())
                conf_loss_logger.add_record(total_step, conf_loss.numpy())
                total_loss_logger.add_record(total_step, total_loss.numpy())

                print("epoch: {}, step: {}, loc_loss={}, conf_loss={}, total_loss={}".format(
                    epoch, step, loc_loss.numpy(), conf_loss.numpy(), total_loss.numpy()))

                if total_step % 100 == 0:
                    fluid.dygraph.save_dygraph(yolact.state_dict(), cfg.model_file)
                    fluid.dygraph.save_dygraph(optimizer.state_dict(), cfg.model_file)


if __name__ == '__main__':
    main()
