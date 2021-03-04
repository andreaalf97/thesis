import numpy as np
import torch
import torchvision.transforms as T
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


class Hue(object):
    def __init__(self, prob=0.5, hue=0.25):
        self.prob = prob
        self.b = T.ColorJitter(hue=hue)

    def __call__(self, item: tuple) -> tuple:
        img, target = item
        if random.random() < self.prob:
            img = self.b(img)
        return img, target


class RealGatesDS(torch.utils.data.Dataset):

    std_transforms = T.Compose([
        ToTensor(),
        Resize((256, 256)),
        AddGaussianNoise(prob=0.05),
        Hue(prob=0.1)
    ])

    def __init__(self, dataset_path, pkl_path, image_set='train', transform=None, mask_rcnn=False):
        assert isinstance(dataset_path, str)
        assert isinstance(pkl_path, (str, list))
        assert image_set in ('train', 'val')

        image_set = 'test' if 'val' in image_set else image_set

        print("[RG DATASET] Initializing Real Gates dataset")
        self.dataset_path = dataset_path
        self.transform = transform if transform is not None else self.std_transforms
        self.mask_rcnn = mask_rcnn

        if isinstance(pkl_path, str):
            self.df = pd.read_pickle(pkl_path)
        else:
            assert len(pkl_path) > 1
            self.df = pd.read_pickle(pkl_path[0])
            for path in pkl_path[1:]:
                tmp_df = pd.read_pickle(path)
                self.df = self.df.append(tmp_df, ignore_index=True)

        self.df = self.df[self.df['split'] == image_set]
        print(f"[RG DATASET] Loaded {len(self.df)} images for {image_set} split")

    def __len__(self):
        return len(self.df)

    def get_height_and_width(self, index):
        return [int(i) for i in self.df.iloc[index]['img_shape']]

    def __getitem__(self, index):

        row = self.df.iloc[index]

        img_path = str(row['img_path'])
        img = Image.open(join(
            self.dataset_path,
            img_path
        ))
        img_id = int(img_path.split('/')[1].replace('.jpg', ''))

        height, width = [int(i) for i in row['img_shape']]
        img_size = [height, width]

        if not self.mask_rcnn:
            gate_coord = row['gates_rel']
            bbox_coord = row['bnd_boxes_rel']
            areas = row['areas_rel']
        else:
            gate_coord = row['gates_abs']
            bbox_coord = row['bnd_boxes_abs']
            areas = row['areas']

            masks = []
            for gate in gate_coord:
                mask_img = Image.new('L', (width, height), 0)
                ImageDraw.Draw(mask_img).polygon([
                    (gate[0], gate[1]),
                    (gate[2], gate[3]),
                    (gate[4], gate[5]),
                    (gate[6], gate[7]),
                ], outline=1, fill=1)
                mask_img = torch.tensor(np.array(mask_img), dtype=torch.int8)
                masks.append(mask_img)
            masks = torch.stack(masks)

        if self.mask_rcnn:
            target = {
                'boxes': torch.tensor(bbox_coord, dtype=torch.float32),
                'gates': torch.tensor(gate_coord, dtype=torch.float32),
                'masks': masks,
                'labels': torch.tensor([1 for _ in range(len(bbox_coord))], dtype=torch.int64),
                'image_id': torch.tensor(img_id, dtype=torch.int64),
                'area': torch.tensor(areas, dtype=torch.float32),
                'iscrowd': torch.tensor([0 for _ in range(len(bbox_coord))], dtype=torch.int64),
                'orig_size': torch.tensor(img_size, dtype=torch.int64),
                'size': torch.tensor(img_size, dtype=torch.int64),
                'isrelative': torch.tensor([False], dtype=torch.bool)
            }
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

        if self.transform:
            img, target = self.transform((img, target))

        return img, target

if __name__ == '__main__':

    # ds = RealGatesDS(
    #     "/home/andreaalf/Documents/thesis/datasets/gate_full_sample",
    #     image_set='train',
    #     mask_rcnn=False
    # )

    ds = RealGatesDS(
        "/home/andreaalf/Documents/thesis/datasets/gate_samples",
        "/home/andreaalf/Documents/thesis/datasets/basement.pkl",
        mask_rcnn=True
    )

    index = random.choice(range(len(ds)))
    img, target = ds[index]

    plt.imshow(img.cpu().permute(1, 2, 0))
    h, w = target['size']
    bnd_box = target['boxes']
    # gates = target['gates']
    # masks = target['masks']
    for box in bnd_box:
        plt.scatter([box[0], box[2]], [box[1], box[3]])
        # plt.scatter([box[2]*w, box[4]*w, box[6]*w], [box[3]*h, box[5]*h, box[7]*h])
        # plt.scatter([box[0]*w], [box[1]*h])
    plt.show()
