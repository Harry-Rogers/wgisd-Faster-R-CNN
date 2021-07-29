# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import numpy as np
import torch
from PIL import Image

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
        self.img_folder= img_folder
        self.xml_folder = xml_folder

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, self.img_folder, self.imgs[idx])
        mask_path = os.path.join(self.root, self.xml_folder, self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        #print(mask_path)
        #print(img_path)
        #print(idx)
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        boxes_to_send = []
        xml_content = XMLHandler(mask_path)
        boxes = xml_content.return_boxes_class_as_dict()
        #print(boxes)
        objs=0
        for box_index in boxes:
            box = boxes[box_index]
            objs = len(boxes)
            xmin = int(box['xmin'])
            ymin = int(box['ymin'])
            xmax = int(box['xmax'])
            ymax = int(box['ymax'])
            boxes_to_send.append([xmin, ymin, xmax, ymax])
        #print(objs)
        boxes = torch.as_tensor(boxes_to_send, dtype=torch.float32)
        labels = torch.ones((objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        
        

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        #print(area)
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
    model = QAT_FASTER_RCNN.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
    # Fuse layers
    for m in model.modules():
        print(model.modules)
        if type(m) == ['backbone.boday.ConvBNActivation']:
            modules_to_fuse = ['0', '1']
            if type(m[2]) == nn.ReLU:
                modules_to_fuse.append('2')
            torch.quantization.fuse_modules(m, modules_to_fuse, inplace=True)
        elif type(m) == ['backbone.body.QuantizableSqueezeExcitation']:
            torch.quantization.fuse_modules(m, ['fc1', 'relu'], inplace=True)
        elif type(m) == ['backbone.body.QuantizableInvertedResidual']:
            for idx in range(len(m.block)):
                if type(m.block[idx]) == nn.Conv2d:
                    torch.quantization.fuse_modules(m.block, [str(idx), str(idx + 1)], inplace=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def interp(model):
    transform = transforms.Compose([
 transforms.Resize(256),
 transforms.CenterCrop(224),
 transforms.ToTensor()
])
    transform_normalize = transforms.Normalize(
     mean=[0.485, 0.456, 0.406],
     std=[0.229, 0.224, 0.225]
 )

    model.eval()
    model.to('cpu:0')
    img = Image.open('test/imgs/CDY_2015.jpg')
    
    transformed_img = transform(img)
    
    input = transform_normalize(transformed_img)
    input = input.unsqueeze(0)
    
    output = model(input)
    print(output)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)
    
    integrated_gradients = IntegratedGradients(model)
    attributions_ig = integrated_gradients.attribute(input, target=pred_label_idx, n_steps=200)
    default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                 [(0, '#ffffff'),
                                                  (0.25, '#000000'),
                                                  (1, '#000000')], N=256)

    _ = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1,2,0)),
                                 np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                 method='heat_map',
                                 cmap=default_cmap,
                                 show_colorbar=True,
                                 sign='positive',
                                 outlier_perc=1)



def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = GerpsFinder('./', get_transform(train=True), img_folder='train/imgs/', xml_folder='train/annos/')
    dataset_test = GerpsFinder('./', get_transform(train=False),img_folder='test/imgs/', xml_folder='test/annos/')


   

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
    
    #interp(model)
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)
    
    model_dir = "saved_models"
    model_filename = "tv-training-Mob.pt"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.jit.save(torch.jit.script(model), model_filepath)
    
    fused_model = copy.deepcopy(model)

    torch.backends.quantized.engine = 'fbgemm'
    quantization_config = torch.quantization.get_default_qconfig('fbgemm')
    fused_model.qconfig = quantization_config
    torch.quantization.prepare_qat(fused_model, inplace=True)
    
    for epoch in range(num_epochs):
        train_one_epoch(fused_model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
    evaluate(fused_model, data_loader_test, device=device) 
    fused_model.to('cpu:0')

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
