import os.path
import time

import numpy as np
import torch
from PIL import ImageFont, ImageDraw, Image
from torch.utils.data import DataLoader

from self_train_fasterrcnn.Dataset import MyDataSet
from self_train_fasterrcnn.coco_eval import CocoEvaluator
from self_train_fasterrcnn.coco_utils import get_coco_api_from_dataset
from self_train_fasterrcnn.engine import _get_iou_types
import utils
batch_size = 2


def drawImage(image, labels_, boxes, color):
    font = ImageFont.truetype(font='/data/kile/other/yolov3/font/FiraMono-Medium.otf',
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300
    for i in zip(labels_, boxes):
        label = i[0].cpu().numpy()
        box = i[1].cpu().detach().numpy()
        left, top, right, bottom = box
        top = int(top)
        left = int(left)
        bottom = int(bottom)
        right = int(right)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(str(label), font)

        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=color)
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=(128, 0, 128))
        draw.text(text_origin, str(label), fill=(0, 0, 0), font=font)
        del draw
    return image




model = torch.load(r"G:\python\2022\04\faster-rcnn-pytorch\self_train_fasterrcnn\temp.pth").cuda().eval()
print(model)
root = r'./1.txt'
image_save_path = r"./result/01"
if not os.path.exists(image_save_path):
    os.makedirs(image_save_path)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


dataset_ = MyDataSet(root, showImagePath=True)
data_loader = DataLoader(
    dataset_, batch_size=batch_size, shuffle=True,  # num_workers=4,
    collate_fn=utils.collate_fn)


n_threads = torch.get_num_threads()
# FIXME remove this and make paste_masks_in_image run on the GPU
torch.set_num_threads(1)
cpu_device = torch.device("cpu")
metric_logger = utils.MetricLogger(delimiter="  ")
header = "Test:"


for images, targets, image_paths in metric_logger.log_every(data_loader, 1, header):
    print(targets)
    images = list(img.to(device) for img in images)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model_time = time.time()
    outputs = model(images)
    outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

    for i in range(batch_size):
        labels_ = targets[i]["labels"]
        boxes_ = targets[i]["boxes"]

        image = drawImage(Image.open(image_paths[i]), labels_, boxes_, (150, 0, 0))

        labels_pre = outputs[i]["labels"]
        boxes_pre = outputs[i]["boxes"]
        image = drawImage(image, labels_pre, boxes_pre, (0,150,0))
        image.save(os.path.join(image_save_path, os.path.basename(image_paths[i])))

    model_time = time.time() - model_time

    res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
    evaluator_time = time.time()
    evaluator_time = time.time() - evaluator_time
