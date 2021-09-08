# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import numpy as np
import torch
from PIL import Image

from matplotlib import pyplot as plt


import copy
import torch.nn as nn

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import torchvision
from engine import train_one_epoch, evaluate
import utils
import transforms as T

from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Dict
import QAT_FASTER_RCNN

from torchvision import transforms

import glob
import time

import collections
import PIL.ImageDraw as ImageDraw
import random
import PIL.ImageFont as ImageFont

import torch.nn.utils.prune as prune
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchsummary import summary


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
        boxes_to_send = []
        xml_content = XMLHandler(mask_path)
        boxes = xml_content.return_boxes_class_as_dict()
        objs = 0
        for box_index in boxes:
            box = boxes[box_index]
            objs = len(boxes)
            xmin = int(box['xmin'])
            ymin = int(box['ymin'])
            xmax = int(box['xmax'])
            ymax = int(box['ymax'])
            boxes_to_send.append([xmin, ymin, xmax, ymax])
    
        boxes = torch.as_tensor(boxes_to_send, dtype=torch.float32)
        labels = torch.ones((objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

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

from torchvision.ops import misc as misc_nn_ops

class QuantizedModel(nn.Module):
    def __init__(self, model_fp32):

        super(QuantizedModel, self).__init__()

        self.quant = torch.quantization.QuantStub()
        self.model_fp32 = model_fp32
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):

        x = self.quant(x)
        x = self.model_fp32(x)
        x = self.dequant(x)
        return x



def get_model_instance_segmentation(num_classes):
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.rpn import AnchorGenerator
    
    # load a pre-trained model for classification and return
    #242 MB for v3 large
    #329MB for v2?
    # only the features
    backbone = torchvision.models.quantization.mobilenet_v3_large(pretrained=True, quantize=False).features

    backbone = QuantizedModel(backbone)
    # FasterRCNN needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280
    # so we need to add it here
    backbone.out_channels = 960
    
    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))
    
    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=4,
                                                    sampling_ratio=2)
    
    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone,
                       num_classes=2,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    
    return model


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
        img = list(img)
        model.eval()
        with torch.no_grad():
            since = time.time()
            predictions = model(img)
            print('{} Time:{}s'.format(i, time.time() - since))
            predictions = list(predictions[1])
            predictions = predictions[0]
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
            
def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def main():
    
    #model = torch.jit.load('./saved_models/tv-training-QAT-Mob.pt', map_location="cpu:0")
    #print(model)
    #time.sleep(10)
    random_seed = 0
    set_random_seeds(random_seed=random_seed)

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
        dataset, batch_size=1, shuffle=True, num_workers=0,
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

    # 25 epochs
    num_epochs = 25

    for epoch in range(num_epochs):
    #    # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10,
                        global_pruning=True, type_prune="", conv2d_prune_amount=0, linear_prune_amount=0, data_loader_test=data_loader_test)
        for i in range(len(params)):
            print(torch.topk(params[i].grad,1))
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
    remove_parameters(model=model)
    coco_eval, metric_logger = evaluate(model, data_loader_test, device=device)


    model_dir = "saved_models"
    model_filename = "Mob.pt"
    
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.jit.save(torch.jit.script(model), model_filepath)

    fused_model = copy.deepcopy(model)
    for m in fused_model.modules():
        #print(model.modules)
        if type(m) == ['backbone.body.ConvBNActivation']:
            modules_to_fuse = ['0', '1']
            if type(m[2]) == nn.ReLU:
                modules_to_fuse.append('2')
            torch.quantization.fuse_modules(m, modules_to_fuse, inplace=True)
        elif type(m) == ['backbone.body.QuantizableSqueezeExcitation']:
            torch.quantization.fuse_modules(m, ['fc1', 'relu'], inplace=True)
        elif type(m) == ['backbone.body.QuantizableInvertedResidual']:
            for idx in range(len(m.block)):
                if type(m.block[idx]) == nn.Conv2d:
                    torch.quantization.fuse_modules(
                        m.block, [str(idx), str(idx + 1)], inplace=True)

    
    del model
    torch.cuda.empty_cache()
    
    fused_model.to(device)
    print(fused_model)
    
    
    
    
    torch.backends.quantized.engine = 'qnnpack'
    quantization_config = torch.quantization.get_default_qconfig('qnnpack')
    fused_model.qconfig = quantization_config

    fused_model.rpn.qconfig = None
    fused_model.roi_heads.qconfig = None
    
    print(fused_model.qconfig)
    torch.quantization.prepare_qat(fused_model, inplace=True)
    
    for epoch in range(num_epochs):
        train_one_epoch(fused_model, optimizer, data_loader, device, epoch, print_freq=50,
                        type_prune="structured", conv2d_prune_amount=1, linear_prune_amount=1)
        # update the learning rate
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        remove_parameters(model=fused_model)
        coco_eval, metric_logger = evaluate(fused_model, data_loader_test, device=device)
        # evaluate on the test dataset
    remove_parameters(model=fused_model)
    #evaluate(fused_model, data_loader_test, device=device)
    
    fused_model.to('cpu:0')

    fused_model = torch.quantization.convert(fused_model, inplace=True)
    
    fused_model = torch.quantization.quantize_dynamic(fused_model,
                                        {torch.nn.Linear},
                                        dtype=torch.qint8)
    
    model_dir = "saved_models"
    model_filename = "Mob-QAT-DYNAM-S.pt"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.jit.save(torch.jit.script(fused_model), model_filepath)
    print(fused_model)
    
    visual_test(fused_model,"Mob-QAT-DYNAM-S", device="cpu", thresh=0.6)
    
    
    print("That's it!")
    
if __name__ == "__main__":
    main()
