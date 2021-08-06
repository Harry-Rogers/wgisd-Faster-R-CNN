# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import numpy as np
import torch
from PIL import Image

import cv2
from matplotlib import pyplot as plt

import sys

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

import GA


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
    

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)




def getmasks(model):
    masks = []
    print(model.modules)
    #time.sleep(20)
    masks.append(model.backbone.body.conv1.weight)
    masks.append(model.backbone.body.layer1[0].conv1.weight)
    #masks.append(model.backbone.body.layer2.conv1.weight)
    #masks.append(model.backbone.body.layer1.conv1.weight)
    #masks.append(model.backbone.body.layer1.conv1.weight)
    #masks.append(model.backbone.body.layer1.conv1.weight)
    #masks.append(model.backbone.body.layer1.conv1.weight)
    #masks.append(model.backbone.body.layer1.conv1.weight)
    #print(masks)
    #print(type(masks))
    #model.backbone.
    #    masks.append(child)
    #for i in model.backbone.named_parameters:
    #    print(i)
    
        
    return masks
    

def make_population(n, shapeslist):
    population = []

    for _ in range(n):
        gnome = GA.Individual.create_gnome_random(shapeslist)
        population.append(GA.Individual(gnome))

    return population




def main():
    #torch.set_printoptions(threshold=sys.maxsize)

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
        dataset, batch_size=2, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)
 
    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)
    masks = getmasks(model)
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
    num_epochs = 1
    
    popsize = 100
    shapelist = [i.shape for i in masks]
    #print(shapelist)
    population = make_population(popsize, shapelist)
    #print(population)
    maxgenerations = 50
    generation = 1
    #Do x generations
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10,
                        global_pruning=False, conv2d_prune_amount=0, linear_prune_amount=0)
        lr_scheduler.step()
        coco_eval, metric_logger = evaluate(model, data_loader_test, device=device)
        
        while generation < maxgenerations:
                sorted_population = sorted(population, key=lambda x: x.cal_fitness(model, data_loader_test, device), reverse=True)
                #print("here")
                fitpercent = 20
        
                # Perform Elitism, that mean 10% of fittest population
                # goes to the next generation
                
               
                s = 10
                new_generation = sorted_population[:s]
                #print(new_generation)
                # From 20% of fittest population, Individuals
                # will mate to produce offspring
                s = 90
                for _ in range(s):
                    parent1 = np.random.choice(sorted_population[:fitpercent])
                    parent2 = np.random.choice(sorted_population[:fitpercent])
                    child = parent1.mate(parent2)
                    new_generation.append(child)
            
            
                population = new_generation
                print(population)
                #acc = new_generation[0].cal_fitness(model, data_loader_test, device)
                #print("Generation: {}\tAccuracy: {:.4f}, population size: {}".format(generation, acc, len(population)))
                generation += 1 

    
    
    
    
    
    
    
    #masks = getmasks(model)


    # interp(model)
   # for epoch in range(num_epochs):
   #     # train for one epoch, printing every 10 iterations
   #     train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10,
   #                     global_pruning=True, conv2d_prune_amount=0, linear_prune_amount=0)
   #     # update the learning rate
    #    lr_scheduler.step()
   #     coco_eval, metric_logger = evaluate(model, data_loader_test, device=device)

        # evaluate on the test dataset
    #import time
    
    #print(model.backbone.body.conv1.weight)
    #conv1_weight = model.backbone.body.conv1.weight
    #print(len(conv1_weight))
    #print(conv1_weight.shape)
    
    #GAUNStruct(model.backbone.body.conv1, name="weight")
    #print(model.backbone.body.conv1.weight)
    #time.sleep(10)
    coco_eval, metric_logger = evaluate(model, data_loader_test, device=device)
    #Change thresh manually
    
    # GA VALUES
    

    


if __name__ == "__main__":
    main()
