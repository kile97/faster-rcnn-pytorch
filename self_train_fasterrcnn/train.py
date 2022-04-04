import time

import torch
import torchvision
from torch.hub import load_state_dict_from_url
from torch.utils.data import Subset
from torchvision.models import resnet50
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _validate_trainable_layers, _resnet_fpn_extractor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, model_urls, FasterRCNN
from torchvision.ops import FrozenBatchNorm2d

from engine import train_one_epoch, evaluate
import utils
from self_train_fasterrcnn.Dataset import MyDataSet


def get_object_detection_model(num_classes, pretrained, pretrained_backbone):
    trainable_backbone_layers = _validate_trainable_layers(
        True, None, 5, 3
    )

    if pretrained:
        pretrained_backbone = False
    # 获取resnet50_backbone, 可以选自torchvision的resnet50预训练模型或者faster rcnn预训练模型
    backbone = resnet50(pretrained=pretrained_backbone, progress=True, norm_layer=FrozenBatchNorm2d)
    # 用_resnet_fpn_extractor获取Faster的骨干网络
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
    # 配置anchor_sizes 这里选用的是我在之前的yolo v3中kmeans聚类的一些数据，这里会生成len(anchor_sizes) * len(aspect_ratios)个anchor box
    anchor_sizes = ((25,), (35,), (50,), (90,), (120,))
    # anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    model = FasterRCNN(backbone, num_classes, rpn_anchor_generator=rpn_anchor_generator)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["fasterrcnn_resnet50_fpn_coco"], progress=True)
        model.load_state_dict(state_dict)
        overwrite_eps(model, 0.0)
    return model

root = r'/data/kile/other/yolov3/data_set_kile/data_txt/enhance_train_train.txt'

# 设置训练设备
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 2类
num_classes = 2
# 设置数据集
dataset_ = MyDataSet(root)

# split the dataset in train and test set
# 我的数据集一共有492张图，差不多训练验证4:1
indices = torch.randperm(len(dataset_)).tolist()
dataset = Subset(dataset_, indices[:-110])
dataset_test = Subset(dataset_, indices[-110:])

# define training and validation data loaders
# 在jupyter notebook里训练模型时num_workers参数只能为0，不然会报错，这里就把它注释掉了
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True,  # num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=2, shuffle=False,  # num_workers=4,
    collate_fn=utils.collate_fn)

# # get the model using our helper function
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=num_classes,
#                                                              pretrained_backbone=True)  # 或get_object_detection_model(num_classes)
model = get_object_detection_model(3,False,True)
# model = FasterRCNN()



# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]

# SGD
optimizer = torch.optim.SGD(params, lr=0.0003,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler
# cos学习率
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)

# let's train it for   epochs
num_epochs = 31

for epoch in range(num_epochs):
    #train for one epoch, printing every 10 iterations
    #engine.py的train_one_epoch函数将images和targets都.to(device)了
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1)

    # update the learning rate
    lr_scheduler.step()

    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

    print('')
    print('==================================================')
    print('')
    torch.save(model, f"./temp/temp_{epoch}_{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}.pth")

print("That's it!")

