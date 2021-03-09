from datasets import get_mask_rcnn_dataset

import time
import torch
import torchvision
import numpy as np
import random
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import math
import sys
import baseline

import matplotlib.pyplot as plt


def show_batch(images: list, targets: list, show_mask=False) -> None:
    for img, target in zip(images, targets):
        plt.imshow(img.cpu().permute(1, 2, 0))

        for bnd_box in target['boxes']:
            print(bnd_box)
            x_min = bnd_box[0].cpu()
            y_min = bnd_box[1].cpu()
            x_max = bnd_box[2].cpu()
            y_max = bnd_box[3].cpu()
            plt.scatter(
                [x_min, x_min, x_max, x_max],
                [y_min, y_max, y_min, y_max],
                s=20
            )

        plt.show()

        if show_mask:
            for mask in target['masks']:
                plt.imshow(mask.cpu())
                plt.title(mask.shape)
                plt.show()


def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
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
    return tuple(zip(*data))


def test_forward():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    dataset = get_mask_rcnn_dataset('/home/andreaalf/Documents/thesis/datasets/gate')
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=False, num_workers=4,
        collate_fn=collate)
    # For Training
    images, targets = next(iter(data_loader))
    images = list(image for image in images)
    targets = [{k: v for k, v in t.items()} for t in targets]
    output = model(images, targets)  # Returns losses and detections
    print("OUT:", output)
    # For inference
    model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = model(x)  # Returns predictions
    print("EVAL:", predictions)


if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    print("RUNNING ON", device)

    num_classes = 2

    seed = 43
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    path = "/tudelft.net/staff-bulk/ewi/insy/VisionLab/yanconglin/dataset/gate_samples"
    pkl_path = "/home/nfs/andreaalfieria/STD_TRAIN_daylight15k_irosFrontal.pkl"
    # path = "/home/andreaalf/Documents/thesis/datasets/gate_samples"
    # pkl_path = "/home/andreaalf/Documents/thesis/datasets/normalized_train_8000imgs.pkl"

    #############################################
    # EVALUATION PARAMETERS
    # Comment this out for no evaluation
    # eval_model = "/home/nfs/andreaalfieria/thesis/detr/tmp/maskrcnn_uniform8000_100epochs.pth"
    eval_model = ""
    # eval_model = "/home/andreaalf/Documents/thesis/detr/results/baseline_comparison/maskrcnn_uniform8000_100epochs.pth"
    eval_pkl_path = "/home/nfs/andreaalfieria/normalized_test_2000imgs.pkl"
    # eval_pkl_path = "/home/andreaalf/Documents/thesis/datasets/normalized_test_2000imgs.pkl"
    save_results_to = "/home/nfs/andreaalfieria/thesis/detr/tmp/EVAL_maskrcnn_uniform8000_100epochs_Gaussian.pkl"

    if eval_model != "":
        baseline.evaluate(
            model=get_instance_segmentation_model(num_classes),
            pkl_path=eval_pkl_path,
            pretrained_model=eval_model,
            ds_func=get_mask_rcnn_dataset,
            ds_path=path,
            save_results_to=save_results_to,
            device=device
        )
        exit(0)
    #############################################

    save_model_to = "/home/nfs/andreaalfieria/thesis/detr/tmp/maskrcnn_STD_100epochs.pth"
    # save_model_to = ""
    num_epochs = 100
    batch_size = 8
    drop_lr_after = 80
    learning_rate = 0.005
    # learning_rate = 1e-4

    #############################################
    ds = get_mask_rcnn_dataset(
        path,
        pkl_path=pkl_path
    )

    dataset_size = len(ds)
    epoch_iterations = math.ceil(dataset_size / batch_size)

    data_loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=4,
        collate_fn=collate)

    #############################################

    model = get_instance_segmentation_model(num_classes)
    model.to(device)

    #############################################

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate,
                                momentum=0.9, weight_decay=0.0005)

    # The scheduler drops the learning rate by 10 every 80 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=drop_lr_after,
                                                   gamma=0.1)

    ex_times = []
    load_times = []

    print("####################################")
    print("PARAMS:")
    print("path", path)
    print('pkl_path', pkl_path)
    print("save_model_to", save_model_to)
    print("num_epochs", num_epochs)
    print("batch_size", batch_size)
    print("learning_rate", learning_rate)
    print("drop_lr_after", drop_lr_after)
    print("####################################")
    print("Start training...")
    model.train()
    for epoch in range(num_epochs):
        start = time.time()
        print("EPOCH", epoch)
        mean_loss = 0.0
        num_losses = 0
        iteration = 0

        start_loader = time.time()
        for images, targets in data_loader:
            load_times.append(time.time()-start_loader)

            images = list(i.to(device) for i in images)
            targets = [{k: v.to(device) for k, v in dictionary.items()} for dictionary in targets]

            # show_batch(images, targets, show_mask=True)
            # exit(0)

            loss_dict = model(images, targets)
            # print('\n'.join([str(k) + ' --> ' + str(v) for k, v in loss_dict.items()]))

            loss = sum(l for l in loss_dict.values())
            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mean_loss += loss.item()
            num_losses += 1
            if iteration % int(epoch_iterations/10) == 0:
                print(f"[Epoch {epoch}] {iteration}/{epoch_iterations}, LR:{lr_scheduler.get_last_lr()} --> Loss: {mean_loss/num_losses}")
                mean_loss = 0.0
                num_losses = 0
            iteration += 1
            start_loader = time.time()

        lr_scheduler.step()

        end = time.time()
        print(f"{end-start}s for this epoch")
        ex_times.append(end-start)

        if save_model_to != "":
            print(f"Saving backup of epoch {epoch} at {save_model_to.replace('.pth', '_checkpoint.pth')}")
            torch.save(model.state_dict(), save_model_to.replace('.pth', '_checkpoint.pth'))
            print("SAVED MODEL at", save_model_to.replace('.pth', '_checkpoint.pth'))

    print("FINISHED TRAINING")
    print("Average training time: %.4f s/epoch" % (sum(ex_times)/num_epochs))
    print("Average sample loading time: %.4f s/batch" % (sum(load_times)/len(load_times)))
    if save_model_to != "":
        torch.save(model.state_dict(), save_model_to)
        print("SAVED MODEL at", save_model_to)
