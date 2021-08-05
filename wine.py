# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import numpy as np
import torch
from PIL import Image

import cv2
from matplotlib import pyplot as plt

import io

import copy
import torch.nn as nn

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from engine import train_one_epoch, evaluate
import utils
import transforms as T

from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Dict
import QAT_FASTER_RCNN
from prune_utils import evaluate_model, create_classification_report

from matplotlib.colors import LinearSegmentedColormap
from torchvision import transforms
from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
import torch.nn.functional as F
import glob
import time

import collections
import PIL.ImageDraw as ImageDraw
import random
import PIL.ImageFont as ImageFont

import torch.nn.utils.prune as prune

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class XMLHandler:
    def __init__(self, xml_path: str or Path):
        self.xml_path = Path(xml_path)
        self.root = self.__open()

    def __open(self):
        with self.xml_path.open() as opened_xml_file:
            self.tree = ET.parse(opened_xml_file)
            return self.tree.getroot()

    def return_boxes_class_as_dict(self) -> Dict[int, Dict]:
        """
        Returns Dict with class name and bounding boxes.
        Key number is box number
        :return:
        """

        boxes_dict = {}
        for index, sg_box in enumerate(self.root.iter('object')):
            boxes_dict[index] = {"name": sg_box.find("name").text,
                                 "xmin": int(sg_box.find("bndbox").find("xmin").text),
                                 "ymin": int(sg_box.find("bndbox").find("ymin").text),
                                 "xmax": int(sg_box.find("bndbox").find("xmax").text),
                                 "ymax": int(sg_box.find("bndbox").find("ymax").text)}

        return boxes_dict


class GerpsFinder(object):
    def __init__(self, root, transforms, img_folder, xml_folder):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, img_folder))))
        self.masks = list(sorted(os.listdir(os.path.join(root, xml_folder))))
        self.img_folder = img_folder
        self.xml_folder = xml_folder

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, self.img_folder, self.imgs[idx])
        mask_path = os.path.join(self.root, self.xml_folder, self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # print(mask_path)
        # print(img_path)
        # print(idx)
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        boxes_to_send = []
        xml_content = XMLHandler(mask_path)
        boxes = xml_content.return_boxes_class_as_dict()
        # print(boxes)
        objs = 0
        for box_index in boxes:
            box = boxes[box_index]
            objs = len(boxes)
            xmin = int(box['xmin'])
            ymin = int(box['ymin'])
            xmax = int(box['xmax'])
            ymax = int(box['ymax'])
            boxes_to_send.append([xmin, ymin, xmax, ymax])
        # print(objs)
        boxes = torch.as_tensor(boxes_to_send, dtype=torch.float32)
        labels = torch.ones((objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # print(area)
        # suppose all instances are not crowd
        iscrowd = torch.zeros((objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_model_instance_segmentation(num_classes):
    model = QAT_FASTER_RCNN.fasterrcnn_resnet50_fpn(pretrained=True)
    print(model)
    fused_model = torch.quantization.fuse_modules(model,[["backbone.body.conv1", "backbone.body.relu"]], inplace=True)
    for module_name, module in fused_model.named_children():
        print(module_name)
        if "layer" in module_name:
            for basic_block_name, basic_block in module.named_children():
                torch.quantization.fuse_modules(basic_block, [['conv1', 'relu1'], ['conv2', 'relu2'], ['conv3']], inplace=True)
                for sub_block_name, sub_block in basic_block.named_children():
                    if sub_block_name == "downsample":
                        torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)

    # get number of input features for the classifier
    in_features = fused_model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    fused_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return fused_model

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def remove_parameters(model):

    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass
        elif isinstance(module, torch.nn.Linear):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass

    return model


STANDARD_COLORS = [
    'Pink', 'Green', 'SandyBrown',
    'SeaGreen',  'Silver', 'SkyBlue', 'White',
    'WhiteSmoke', 'YellowGreen'
]


def filter_low_thresh(boxes, scores, classes, category_index, thresh, box_to_display_str_map, box_to_color_map, col):
    for i in range(boxes.shape[0]):
        if scores[i] > thresh:
            box = tuple(boxes[i].tolist())  # numpy -> list -> tuple
            if classes[i] in category_index.keys():
                class_name = category_index[classes[i]]
            else:
                class_name = 'N/A'
            display_str = str(class_name)
            display_str = '{}: {}%'.format(display_str, int(100 * scores[i]))
            box_to_display_str_map[box].append(display_str)
            box_to_color_map[box] = STANDARD_COLORS[col]
        else:
            break  # Scores have been sorted


def draw_text(draw, box_to_display_str_map, box, left, right, top, bottom, color):
    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()

    display_str_heights = [font.getsize(ds)[1] for ds in box_to_display_str_map[box]]
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    for display_str in box_to_display_str_map[box][::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)], fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                  display_str,
                  fill='black',
                  font=font)
        text_bottom -= text_height - 2 * margin


def draw_box(image, boxes, classes, scores, category_index, thresh=0.5, line_thickness=20):
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
    
    col = int(random.random() * len(STANDARD_COLORS))
    filter_low_thresh(boxes, scores, classes, category_index, thresh, box_to_display_str_map, box_to_color_map, col)

    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    for box, color in box_to_color_map.items():
        xmin, ymin, xmax, ymax = box
        (left, right, top, bottom) = (xmin * 1, xmax * 1,
                                      ymin * 1, ymax * 1)
        draw.line([(left, top), (left, bottom), (right, bottom),
                   (right, top), (left, top)], width=line_thickness, fill=color)
        draw_text(draw, box_to_display_str_map, box, left, right, top, bottom, color)

def visual_test(model, model_name, device, thresh):
        # read class_indict
    category_index = {"Grape": 1}
    
    origin_list = glob.glob('./test/imgs/*.jpg')
    
    preds = []
    model.to(device)
    for i, img_name in enumerate(origin_list):
        # load image
        original_img = Image.open(img_name)
    
        # from pil image to tensor, do not normalize image
        data_transform = transforms.Compose([transforms.ToTensor()])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)
    
        model.eval()
        with torch.no_grad():
            since = time.time()
            predictions = model(img.to(device))[0]
            print('{} Time:{}s'.format(i, time.time() - since))
            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()
    
            draw_box(original_img,
                     predict_boxes,
                     predict_classes,
                     predict_scores,
                     category_index,
                     thresh=thresh,
                     line_thickness=15)
            plt.imshow(original_img)
            plt.savefig(str(i) + model_name + ".jpg")
            plt.show()
            
    
            predict = ""
            for box, score in zip(predict_boxes, predict_scores):
                str_box = ""
                box[2] = box[2] - box[0]
                box[3] = box[3] - box[1]
                for b in box:
                    str_box += str(b) + ' '
                predict += str(score) + ' ' + str_box
            preds.append(predict)


def main():
    
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = GerpsFinder('./', get_transform(train=True), img_folder='train/imgs/', xml_folder='train/annos/')
    dataset_test = GerpsFinder('./', get_transform(train=False), img_folder='test/imgs/', xml_folder='test/annos/')

   # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)
 
    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 25

    # interp(model)
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10,
                        global_pruning=True, conv2d_prune_amount=0, linear_prune_amount=0)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
    remove_parameters(model=model)
    coco_eval, metric_logger = evaluate(model, data_loader_test, device=device)
    #Change thresh manually
    visual_test(model, "res_net", device, thresh=0.6)
    
    # GA VALUES
    AP_1 = metric_logger[91:97]
    AP_2 = metric_logger[170:176]
    AP_3 = metric_logger[251:257]
    AP_4 = metric_logger[331:337]#Negative can ignore also no small images in data
    AP_5 = metric_logger[412:418]
    AP_6 = metric_logger[492:498]
    
    AR_1 = metric_logger[572:578]
    AR_2 = metric_logger[652:658]
    AR_3 = metric_logger[732:738]
    AR_4 = metric_logger[812:818]#Negative can ignore
    AR_5 = metric_logger[892:898]
    AR_6 = metric_logger[973:979]

    model_dir = "saved_models"
    model_filename = "tv-training-Mob.pt"
    visual_test(model, "Normal", 'cuda')
    
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.jit.save(torch.jit.script(model), model_filepath)

    fused_model = copy.deepcopy(model)
    
    del model
    torch.cuda.empty_cache()
    
    fused_model.to(device)
    
    torch.backends.quantized.engine = 'fbgemm'
    quantization_config = torch.quantization.get_default_qconfig('fbgemm')
    fused_model.qconfig = quantization_config
    torch.quantization.prepare_qat(fused_model, inplace=True)
    
    for epoch in range(num_epochs):
        train_one_epoch(fused_model, optimizer, data_loader, device, epoch, print_freq=10,
                        global_pruning=True, conv2d_prune_amount=0, linear_prune_amount=0)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
    remove_parameters(model=fused_model)
    evaluate(fused_model, data_loader_test, device=device)

    fused_model.to('cpu:0')
    
    visual_test(fused_model, "QAT_AWARE_RES", 'cpu', thresh=0.6)

    # Using high-level static quantization wrapper
    # The above steps, including torch.quantization.prepare, calibrate_model, and torch.quantization.convert, are also equivalent to
    # quantized_model = torch.quantization.quantize_qat(model=quantized_model, run_fn=train_model, run_args=[train_loader, test_loader, cuda_device], mapping=None, inplace=False)
    
    fused_model = torch.quantization.convert(fused_model, inplace=True)

    model_dir = "saved_models"
    model_filename = "tv-training-QAT-Mob.pt"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.jit.save(torch.jit.script(fused_model), model_filepath)
    

    
    print("That's it!")


if __name__ == "__main__":
    main()
