import numpy as np
import torch
import torchvision.transforms as T
from os.path import join, isfile
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random
import pandas as pd


CLASSES = {
    '<start>': 0,
    '<polygon>': 1,
    '<end-of-computation>': 2
}

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


class GetSentence(object):
    def __init__(self, seq_order, max_gates=4):
        self.max_gates = max_gates
        self.seq_order = seq_order

    def __call__(self, sample):
        img, target = sample

        max_lenght = self.max_gates + 2

        sequence = []
        start_token = torch.zeros(256)
        start_token[8 + CLASSES['<start>']] = 1
        sequence.append(start_token)

        if self.seq_order in ('lr', 'rl'):
            centers = []
            for polygon in target['boxes']:
                mean_x = polygon[::2].mean().item()
                mean_y = polygon[1::2].mean().item()
                centers.append((mean_x, mean_y))
            centers = {i: c for i, c in enumerate(centers)}

            if 'lr' in self.seq_order:
                centers = sorted(centers.items(), key=lambda x: x[1][0])
            else:
                centers = sorted(centers.items(), key=lambda x: x[1][0], reverse=True)

            for i, center in centers:
                polygon = target['boxes'][i]
                token = torch.zeros(256)
                token[8 + CLASSES['<polygon>']] = 1
                token[:8] = polygon
                sequence.append(token)
        elif self.seq_order in ('ls', 'sl'):
            areas = {i: c.item() for i, c in enumerate(target['area'])}

            print(areas)
            print(areas.items())
            print(
                sorted(areas.items(), key=lambda x: x[1])
            )

            if 'ls' in self.seq_order:
                areas = sorted(areas.items(), key=lambda x: x[1], reverse=True)
            else:
                areas = sorted(areas.items(), key=lambda x: x[1], reverse=False)

            for element in areas:
                polygon = target['boxes'][element[0]]
                token = torch.zeros(256)
                token[8 + CLASSES['<polygon>']] = 1
                token[:8] = polygon
                sequence.append(token)

        while len(sequence) < max_lenght:
            end_computation = torch.zeros(256)
            end_computation[8 + CLASSES['<end-of-computation>']] = 1
            sequence.append(end_computation)

        sequence = torch.stack(sequence)

        target['sequence'] = sequence
        return img, target


def order_y(points: list):
    assert len(points) == 2 and isinstance(points[0], tuple)

    y_s = [points[0][1], points[1][1]]
    y_min_index = y_s.index(min(y_s))
    if y_min_index == 0:
        y_max_index = 1
    else:
        y_max_index = 0
    return points[y_min_index], points[y_max_index]


def reorder(target):
    if not target['isrelative']:
        new_gates = []
        for gate in target['gates']:
            x_s = [gate[0].item(), gate[2].item(), gate[4].item(), gate[6].item()]
            y_s = [gate[1].item(), gate[3].item(), gate[5].item(), gate[7].item()]

            min_x = min(x_s)
            min_x_index = x_s.index(min(x_s))

            left = [(min_x, y_s[min_x_index])]

            del x_s[min_x_index]
            del y_s[min_x_index]

            min_x = min(x_s)
            min_x_index = x_s.index(min(x_s))
            left.append((min_x, y_s[min_x_index]))

            del x_s[min_x_index]
            del y_s[min_x_index]

            right = [
                (x_s[0], y_s[0]),
                (x_s[1], y_s[1])
            ]

            tl, bl = order_y(left)
            tr, br = order_y(right)
            new_gates.append([
                bl[0], bl[1],
                tl[0], tl[1],
                tr[0], tr[1],
                br[0], br[1]
            ])
        target['gates'] = torch.tensor(new_gates, dtype=torch.float32)
    else:
        new_gates = []
        for gate in target['boxes']:
            x_s = [gate[0].item(), gate[2].item(), gate[4].item(), gate[6].item()]
            y_s = [gate[1].item(), gate[3].item(), gate[5].item(), gate[7].item()]

            min_x = min(x_s)
            min_x_index = x_s.index(min(x_s))

            left = [(min_x, y_s[min_x_index])]

            del x_s[min_x_index]
            del y_s[min_x_index]

            min_x = min(x_s)
            min_x_index = x_s.index(min(x_s))
            left.append((min_x, y_s[min_x_index]))

            del x_s[min_x_index]
            del y_s[min_x_index]

            right = [
                (x_s[0], y_s[0]),
                (x_s[1], y_s[1])
            ]

            tl, bl = order_y(left)
            tr, br = order_y(right)
            new_gates.append([
                bl[0], bl[1],
                tl[0], tl[1],
                tr[0], tr[1],
                br[0], br[1]
            ])
        target['boxes'] = torch.tensor(new_gates, dtype=torch.float32)
    return target


class RealGatesDS(torch.utils.data.Dataset):

    folder_codes = {
        "basement_course1": 0,
        "basement_course3": 1,
        "bebop_merge": 2,
        "bebop_merge_distort": 3,
        "cyberzoo": 4,
        "daylight15k": 5,
        "daylight_course1": 6,
        "daylight_course3": 7,
        "daylight_course5": 8,
        "daylight_flight": 9,
        "eth": 10,
        "google_merge_distort": 11,
        "iros2018_course1": 12,
        "iros2018_course3_test": 13,
        "iros2018_course5": 14,
        "iros2018_flights": 15,
        "iros2018_frontal": 16,
        "random_flight": 17
    }

    def __init__(self, dataset_path, pkl_path, image_set='train', transform=None, mask_rcnn=False, seq_order='tb'):
        assert isinstance(dataset_path, str)
        assert isinstance(pkl_path, (str, list))
        assert image_set in ('train', 'val')
        assert seq_order in ('ls', 'sl', 'lr', 'rl', 'random'), f"{seq_order} order not implemented for REAL GATES"

        self.std_transforms = T.Compose([
            ToTensor(),
            # Resize((256, 256)),
            Hue(prob=0.1),
            RandomHorizontalFlip(prob=0.4),
            RandomVerticalFlip(prob=0.4),
            AddGaussianNoise(prob=0.1),
            GetSentence(seq_order)
        ])

        self.val_transform = T.Compose([ToTensor(), GetSentence(seq_order)])

        print("[RG DATASET] Initializing Real Gates dataset")
        print(f"[RG DATASET] Sequence ordering will be: {seq_order}")
        self.dataset_path = dataset_path
        self.transform = transform if transform is not None else self.std_transforms
        self.mask_rcnn = mask_rcnn
        self.seq_order = seq_order

        if 'val' in image_set:
            image_set = 'test'
            self.transform = self.val_transform

        if isinstance(pkl_path, str):
            self.df = pd.read_pickle(pkl_path)
        else:
            assert len(pkl_path) > 1
            self.df = pd.read_pickle(pkl_path[0])
            for path in pkl_path[1:]:
                tmp_df = pd.read_pickle(path)
                self.df = self.df.append(tmp_df, ignore_index=True)

        self.df = self.df[self.df['split'] == image_set]
        self.max_gates = self.df['num_gates'].max()
        print(f"[RG DATASET] Loaded {len(self.df)} images for {image_set} split")

    def __len__(self):
        return len(self.df)

    def get_height_and_width(self, index):
        return [int(i) for i in self.df.iloc[index]['img_shape']]

    def __getitem__(self, index):

        row = self.df.iloc[index]

        img_path = str(row['img_path'])
        folder_name = img_path.split('/')[0]
        img = Image.open(join(
            self.dataset_path,
            img_path
        ))
        img_id = int(img_path.split('/')[1].replace('.jpg', '')) + (100000*self.folder_codes[folder_name])

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

        target = reorder(target)

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
        "/home/andreaalf/Documents/thesis/datasets/STD_TRAIN_daylight15k_irosFrontal.pkl",
        mask_rcnn=False,
        image_set='train',
        seq_order='rl'
    )

    index = random.choice(range(len(ds)))
    # index = 0
    img, target = ds[index]

    plt.imshow(img.cpu().permute(1, 2, 0))
    h, w = target['size']
    sequence = target['sequence']
    # gates = target['gates']
    # masks = target['masks']
    for i, element in enumerate(sequence):
        if element[9] == 1:
            plt.scatter(
                element[:8:2].cpu() * w,
                element[1:8:2].cpu() * h,
                label=i
            )
        # plt.scatter([box[0], box[2]], [box[1], box[3]])
        # plt.scatter([box[2]*w, box[4]*w, box[6]*w], [box[3]*h, box[5]*h, box[7]*h])
        # plt.scatter([box[0]*w], [box[1]*h])
        # plt.scatter([box[0]*w], [box[1]*h], label='0')
        # plt.scatter([box[2]*w], [box[3]*h], label='1')
        # plt.scatter([box[4]*w], [box[5]*h], label='2')
        # plt.scatter([box[6]*w], [box[7]*h], label='3')

    plt.legend()
    plt.title(target['image_id'].item())
    plt.show()
    # for mask in target['masks']:
    #     plt.imshow(mask)
    #     plt.show()
