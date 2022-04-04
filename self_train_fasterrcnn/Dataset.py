import os

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor


class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, txt_path, showImagePath = False):
        self.txt_path = txt_path
        # load all image files, sorting them to ensure that they are aligned
        self.contents = []
        with open(self.txt_path, "r") as txt:
            contents = txt.readlines()
        for str_ in contents:
            if len(str_.split(" ")) > 1:
                self.contents.append(str_.replace("\n", ""))
        self.transforms = ToTensor()
        self.showImagePath = showImagePath

    def __getitem__(self, idx):
        # load images and bbox
        img_path = self.contents[idx].replace("\n","").split(" ")[0]
        img = Image.open(img_path).convert("RGB")

        objects = self.contents[idx].replace("\n", "").split(" ")[1:]
        # get bounding box coordinates
        boxes = []
        labels = []
        for object_ in objects:
            xmin, ymin, xmax, ymax, label = np.int(object_.split(",")[0]), np.int(object_.split(",")[1]), np.int(object_.split(",")[2]), np.int(object_.split(",")[3]), object_.split(",")[4]
            # 获取标签中内容
            labels.append(np.int(label) + 1)  #
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        try:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        except Exception as e:
            raise Exception(f" {img_path} 0{ objects} {boxes} {e}")
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(objects),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # 由于训练的是目标检测网络，因此没有教程中的target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            # 注意这里target(包括bbox)也转换\增强了，和from torchvision import的transforms的不同
            # https://github.com/pytorch/vision/tree/master/references/detection 的 transforms.py里就有RandomHorizontalFlip时target变换的示例
            # img, target = self.transforms(img, target)
            img = self.transforms(img)
        if self.showImagePath:
            return img, target, img_path
        return img, target

    def __len__(self):
        return len(self.contents)
