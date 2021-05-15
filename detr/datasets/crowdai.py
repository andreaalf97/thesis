import numpy as np
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from os.path import join, isfile
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random
import pandas as pd


class ToTensor(object):
    def __init__(self):
        self.transform = T.ToTensor()

    def __call__(self, sample):
        img, target = sample
        img = self.transform(img)
        return img, target


class Resize(object):
    def __init__(self, dim):
        if isinstance(dim, int):
            self.dim = (dim, dim)
        elif isinstance(dim, (tuple, list)):
            self.dim = dim
        else:
            raise Exception(f"Resize expects int, tuple or list as type, received {type(dim)}")
        self.transform = T.Resize(dim)

    def __call__(self, sample):
        img, target = sample
        height, width = target['size']
        new_height, new_width = self.dim
        isrelative = target['isrelative']

        img = self.transform(img)
        if not isrelative:
            bboxes = target['boxes']
            gates = target['gates']

            new_bboxes = []
            new_gates = []
            for bbox, gate in zip(bboxes, gates):
                new_bboxes.append([
                    bbox[0] / width * new_width,
                    bbox[1] / height * new_height,
                    bbox[2] / width * new_width,
                    bbox[3] / height * new_height,
                ])
                new_gates.append([
                    gate[0] / width * new_width,
                    gate[1] / height * new_height,
                    gate[2] / width * new_width,
                    gate[3] / height * new_height,
                    gate[4] / width * new_width,
                    gate[5] / height * new_height,
                    gate[6] / width * new_width,
                    gate[7] / height * new_height,
                ])

            new_masks = []
            for mask in target["masks"]:
                new_mask = self.transform(mask.unsqueeze(0))
                new_masks.append(new_mask.squeeze(0))

            new_bboxes = torch.tensor(new_bboxes, dtype=torch.float32)
            new_gates = torch.tensor(new_gates, dtype=torch.float32)
            new_masks = torch.stack(new_masks)

            target.update({
                'boxes': new_bboxes,
                'gates': new_gates,
                'masks': new_masks,
                'size': torch.tensor([new_height, new_width], dtype=torch.int64)
            })
        else:
            target.update({
                'size': torch.tensor([new_height, new_width], dtype=torch.int64)
            })

        return img, target


class AddGaussianNoise(object):
    def __init__(self, prob=0.2, mean=0., std=0.1):
        self.std = std
        self.mean = mean
        self.prob = prob

    def __call__(self, sample):
        img, target = sample
        if random.random() < self.prob:
            return img + torch.randn(img.size()) * self.std + self.mean, target
        else:
            return img, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob
        self.t = T.RandomHorizontalFlip(p=1.0)

    def __call__(self, item: tuple) -> tuple:
        img, target = item
        if random.random() < self.prob:
            img = self.t(img)

            if not target['isrelative']:
                _, width = target['size']
                new_masks = []
                for mask in target['masks']:
                    new_masks.append(self.t(mask))
                target['masks'] = torch.stack(new_masks)

                new_gates = []
                for gate in target['gates']:
                    new_gates.append([
                        width - gate[6].item(),
                        gate[7].item(),
                        width - gate[4].item(),
                        gate[5].item(),
                        width - gate[2].item(),
                        gate[3].item(),
                        width - gate[0].item(),
                        gate[1].item()
                    ])
                    target['gates'] = torch.tensor(new_gates, dtype=torch.float32)

                new_boxes = []
                for bbox in target['boxes']:
                    new_boxes.append([
                        width - bbox[2].item(),
                        bbox[1].item(),
                        width - bbox[0].item(),
                        bbox[3].item()
                    ])
                target['boxes'] = torch.tensor(new_boxes, dtype=torch.float32)
            else:
                new_boxes = []
                for bbox in target['boxes']:
                    new_boxes.append([
                        1.0 - bbox[6].item(),
                        bbox[7].item(),
                        1.0 - bbox[4].item(),
                        bbox[5].item(),
                        1.0 - bbox[2].item(),
                        bbox[3].item(),
                        1.0 - bbox[0].item(),
                        bbox[1].item()
                    ])
                target['boxes'] = torch.tensor(new_boxes, dtype=torch.float32)

        return img, target


class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob
        self.t = T.RandomVerticalFlip(p=1.0)

    def __call__(self, item: tuple) -> tuple:
        img, target = item
        if random.random() < self.prob:
            img = self.t(img)

            if not target['isrelative']:
                height, _ = target['size']
                new_masks = []
                for mask in target['masks']:
                    new_masks.append(self.t(mask))
                target['masks'] = torch.stack(new_masks)

                new_gates = []
                for gate in target['gates']:
                    new_gates.append([
                        gate[2].item(),
                        height - gate[3].item(),
                        gate[0].item(),
                        height - gate[1].item(),
                        gate[6].item(),
                        height - gate[7].item(),
                        gate[4].item(),
                        height - gate[5].item()
                    ])
                    target['gates'] = torch.tensor(new_gates, dtype=torch.float32)

                new_boxes = []
                for bbox in target['boxes']:
                    new_boxes.append([
                        bbox[0].item(),
                        height - bbox[3].item(),
                        bbox[2].item(),
                        height - bbox[1].item()
                    ])
                target['boxes'] = torch.tensor(new_boxes, dtype=torch.float32)
            else:
                new_boxes = []
                for bbox in target['boxes']:
                    new_boxes.append([
                        bbox[2].item(),
                        1.0 - bbox[3].item(),
                        bbox[0].item(),
                        1.0 - bbox[1].item(),
                        bbox[6].item(),
                        1.0 - bbox[7].item(),
                        bbox[4].item(),
                        1.0 - bbox[5].item()
                    ])
                target['boxes'] = torch.tensor(new_boxes, dtype=torch.float32)

        return img, target


class Hue(object):
    def __init__(self, prob=0.5, hue=0.25):
        self.prob = prob
        self.b = T.ColorJitter(hue=hue)

    def __call__(self, item: tuple) -> tuple:
        img, target = item
        if random.random() < self.prob:
            img = self.b(img)
        return img, target

class CrowdAiDataset(torch.utils.data.Dataset):

    std_transforms = T.Compose([
        ToTensor(),
        # Resize((256, 256)),
        # Hue(prob=0.1),
        # RandomHorizontalFlip(prob=0.4),
        # RandomVerticalFlip(prob=0.4),
        # AddGaussianNoise(prob=0.1)
    ])

    val_transform = T.Compose([ToTensor()])

    def __init__(self, dataset_path, image_set='train', transform=None, mask_rcnn=False):
        assert isinstance(dataset_path, str)
        assert image_set in ('train', 'val', 'test')

        print("[CrowdAI DATASET] Initializing CrowdAI dataset")
        self.dataset_path = dataset_path
        self.transform = transform if transform is not None else self.std_transforms
        self.mask_rcnn = mask_rcnn
        self.image_set = image_set
        self.home_folder = join(dataset_path, image_set)

        if 'train' not in image_set:
            self.transform = self.val_transform

        # self.df_ann = pd.read_pickle(join(self.home_folder, 'annotations.pkl'))
        # self.df_images = pd.read_pickle(join(self.home_folder, 'images.pkl'))

        self.df_ann = pd.read_pickle(join(self.home_folder, 'annotations_test.pkl'))
        self.df_images = pd.read_pickle(join(self.home_folder, 'images_test.pkl'))

        self.max_coords = max(self.df_ann.apply(lambda x: len(x['segmentation'][0]), axis=1))
        print("self.max_coords")
        print(self.max_coords)

        # self.df_ann.loc[self.df_ann['image_id'] == 54062].to_pickle(join(self.home_folder, 'annotations_test.pkl'))
        # self.df_images.loc[self.df_images['file_name'] == '000000054062.jpg'].to_pickle(join(self.home_folder, 'images_test.pkl'))

        print(f"[CrowdAI DATASET] Loaded dataset of {len(self.df_images)} images for {image_set} split")

    def __len__(self):
        return len(self.df_images)

    def get_height_and_width(self, index):
        row = self.df_images.iloc[index]
        return int(row['height']), int(row['width'])

    def __getitem__(self, index):

        row = self.df_images.iloc[index]

        print("----------------------------------------")
        print("self.df_images\n")
        print(row)
        print("----------------------------------------")

        img_path = str(row['file_name'])
        img_id = int(row['id'])
        w, h = int(row['width']), int(row['height'])

        img = Image.open(join(
            self.home_folder,
            'images',
            img_path
        ))

        img_size = [h, w]

        annotations = self.df_ann.loc[self.df_ann['image_id'] == img_id]

        points = [torch.tensor(object, dtype=torch.float32).squeeze(0) for object in list(annotations['segmentation'])]
        points = torch.stack([torch.cat([object, torch.full([self.max_coords - len(object)], -1)]) for object in points])
        points = points.view((points.shape[0], -1, 2))

        mask_x = torch.where(points[:, :, 0] > 0, 1, 0)
        mask_y = torch.where(points[:, :, 1] > 0, 1, 0)

        points[:, :, 0] = points[:, :, 0][mask_x == 1] / w
        print(points)
        exit(0)

        print("----------------------------------------")
        print("annotations\n")
        print(annotations.iloc[0])
        print("----------------------------------------")

        print("----------------------------------------")
        print("points\n")
        print(points.shape)
        print(points)
        print("----------------------------------------")

        exit(0)



        if self.mask_rcnn:
            raise Exception("[CrowdAI Dataset] Loader not ready for MASK R-CNN")
            # target = {
            #     'boxes': torch.tensor(bbox_coord, dtype=torch.float32),
            #     'gates': torch.tensor(gate_coord, dtype=torch.float32),
            #     'masks': masks,
            #     'labels': torch.tensor([1 for _ in range(len(bbox_coord))], dtype=torch.int64),
            #     'image_id': torch.tensor(img_id, dtype=torch.int64),
            #     'area': torch.tensor(areas, dtype=torch.float32),
            #     'iscrowd': torch.tensor([0 for _ in range(len(bbox_coord))], dtype=torch.int64),
            #     'orig_size': torch.tensor(img_size, dtype=torch.int64),
            #     'size': torch.tensor(img_size, dtype=torch.int64),
            #     'isrelative': torch.tensor([False], dtype=torch.bool)
            # }
        else:
            target = {
                'boxes': torch.tensor(gate_coord, dtype=torch.float32),
                'labels': torch.tensor([0 for _ in range(len(bbox_coord))], dtype=torch.int64),
                'image_id': torch.tensor(img_id, dtype=torch.int64),
                'area': torch.tensor(areas, dtype=torch.float32),
                'iscrowd': torch.tensor([0 for _ in range(len(bbox_coord))], dtype=torch.int64),
                'orig_size': torch.tensor(img_size, dtype=torch.int64),
                'size': torch.tensor(img_size, dtype=torch.int64),
                'isrelative': torch.tensor([True], dtype=torch.bool)
            }

        # target = reorder(target)

        if self.transform:
            img, target = self.transform((img, target))

        return img, target

if __name__ == '__main__':

    # ds = RealGatesDS(
    #     "/home/andreaalf/Documents/thesis/datasets/gate_full_sample",
    #     image_set='train',
    #     mask_rcnn=False
    # )

    ds = CrowdAiDataset(
        "/home/andreaalf/Documents/thesis/datasets/crowdai",
        mask_rcnn=False,
        image_set='train'
    )

    index = random.randint(0, len(ds)-1)
    # index = 0
    img, target = ds[index]

    plt.imshow(img.cpu().permute(1, 2, 0))
    h, w = target['size']
    bnd_box = target['boxes']
    # gates = target['gates']
    # masks = target['masks']
    for box in bnd_box:
        # plt.scatter([box[0], box[2]], [box[1], box[3]])
        # plt.scatter([box[2]*w, box[4]*w, box[6]*w], [box[3]*h, box[5]*h, box[7]*h])
        # plt.scatter([box[0]*w], [box[1]*h])
        plt.scatter([box[0]*w], [box[1]*h], label='0')
        plt.scatter([box[2]*w], [box[3]*h], label='1')
        plt.scatter([box[4]*w], [box[5]*h], label='2')
        plt.scatter([box[6]*w], [box[7]*h], label='3')

    plt.legend()
    plt.title(target['image_id'].item())
    plt.show()
    # for mask in target['masks']:
    #     plt.imshow(mask)
    #     plt.show()
