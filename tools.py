import tkinter as tk
import matplotlib
import requests
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import os
import glob
import cv2

from PIL import Image
from torch import nn
from torchvision.models import resnet50

x = 0

class DETRdemo(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)

        # construct positional encodings
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        # propagate through the transformer
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1)).transpose(0, 1)
        
        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_logits': self.linear_class(h), 
                'pred_boxes': self.linear_bbox(h).sigmoid()}

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def detect(im, model, transform):
    img = transform(im).unsqueeze(0)

    assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

    outputs = model(img)

    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.7

    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep], bboxes_scaled

def plot_results(pil_img, prob, boxes,filename,no_of_cars):

    plt.figure(figsize=(10,8))
    plt.imshow(pil_img)

    ax = plt.gca()

    # ax.add_patch(plt.Rectangle((250,320), 3,90,
    #                                fill=False, color=[0.000, 0.447, 0.741], linewidth=3))
    # # ax.add_patch(plt.Rectangle((700,450), 3,300,
    # #                                fill=False, color=[0.850, 0.325, 0.098], linewidth=3))
    

    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
        begin = 0
        
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'

        if cl == 3:
            c = [0.000, 0.447, 0.741]
        elif cl == 4:
            c = [0.850, 0.325, 0.098]
        elif cl == 6:
            c = [0.929, 0.694, 0.125]
        elif cl ==8:
            c = [0.494, 0.184, 0.556]
        else:
            continue

        global x
        
        # if x == 0:
        #     if ymin >= 170 and xmin <= 700:
        #         no_of_cars += 1
        if ymin >= 170:
            no_of_cars += 1 

        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))

    
    ax.text(700, 320, "Current no. of \n Cars in lane 1: \n" + str(no_of_cars), fontsize=15,
            bbox=dict(facecolor='yellow', alpha=0.5))

    ax.text(700, 500, "Current no. of \n Cars in lane 2: \n" + str(no_of_cars), fontsize=15,
            bbox=dict(facecolor='green', alpha=0.5))

    x = 1
    plt.axis('off')
    plt.show()
    plt.savefig('output/' + filename )
    plt.close()
    return no_of_cars 
    
torch.set_grad_enabled(False);

detr = DETRdemo(num_classes=91)
state_dict = torch.hub.load_state_dict_from_url(
    url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
    map_location='cpu', check_hash=True)
detr.load_state_dict(state_dict)
detr.eval();

CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def currentFrame(video, frame, i):
    video.set(cv2.CAP_PROP_POS_MSEC,frame*1000)
    hasFrames,image = video.read()
    if hasFrames:
        cv2.imwrite("images/image"+str(i)+".jpg", image)
    return hasFrames

def video_to_images(video_path):

    files = glob.glob('output/*')
    for f in files:
        os.remove(f)

    files = glob.glob('images/*')
    for f in files:
        os.remove(f)

    vidcap = cv2.VideoCapture(video_path)

    frame = 0
    i = 0
    fps = 0.5 
    finished = currentFrame(vidcap, frame, i)
    
    while finished:
        frame = round((frame + fps), 2)
        i = i + 1
        finished = currentFrame(vidcap, frame, i)
    
    return i

def detect_video(path):

    totalFrames = video_to_images(path)
    no_of_cars = 0

    for file in range(totalFrames):

        no_of_cars = 0
        file_name = 'image' + str(file ) + '.jpg'
        im = Image.open('images/' + file_name)
        scores, boxes = detect(im, detr, transform)
        plot_results(im, scores, boxes, file_name, no_of_cars) 

# detect_video('/home/liam/Documents/uni/visionGroupWork/annotations/MorningImages/236.jpg')
detect_video('/home/liam/Documents/uni/visionGroupWork/annotations/testingVideo/20200404_124215TestTrim.mp4')