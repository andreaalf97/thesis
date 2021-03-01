from datasets import get_mask_rcnn_dataset

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import matplotlib.pyplot as plt


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def collate(data):
    """data is a list of lenght BATCH_SIZE of tuples"""

    imgs = torch.stack([img for img, target in data])
    targets = [t for img, t in data]
    return imgs, targets

if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    path = "/home/andreaalf/Documents/thesis/datasets/gate_full_sample"
    ds = get_mask_rcnn_dataset(path)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    data_loader = torch.utils.data.DataLoader(
        ds, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=collate)
