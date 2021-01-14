from image_gen import display_image, get_ts_image
from detr import Transformer
from data import TSDataset
import torchvision.transforms as T
import torch.utils.data as D
import torch
import matplotlib.pyplot as plt
import time
import datetime

batch_size = 1
lr_backbone = 1e-5
lr = 1e-4
weight_decay = 1e-4
lr_drop = 200

epochs = 5
num_batches_per_epoch = 50

if __name__ == '__main__':



    transform = T.ToTensor()

    dataset = TSDataset(256, 256, transform=transform)

    train_loader = D.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # for image, labels in train_loader:
    #     plt.imshow(image[0].permute(1, 2, 0))
    #     plt.show()
    #     print(len(labels))
    #     break

    model = Transformer(1)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": lr_backbone,
        },
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=lr,
                                  weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_drop)

    start_time = time.time()
    print("START TRAINING")

    for e in range(epochs):
        print("Starting epoch {}".format(e+1))
        for _ in range(num_batches_per_epoch):
            pass

    end_time = time.time()
    print("FINISHED TRAINING")

    total_time = end_time - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Completed {} epochs in {}\\{} batches per epoch'.format(epochs, total_time_str, num_batches_per_epoch))



#
# model = Transformer(2)
# model.eval()
#
# CLASSES = ["N/A", "GATE"]
