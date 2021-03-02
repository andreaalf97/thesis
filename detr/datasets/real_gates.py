import numpy as np
import torch
import torchvision.transforms as T
from os import listdir
from os.path import join, isfile
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import random
import pickle


class ToTensor(object):
    def __init__(self):
        self.t = T.ToTensor()

    def __call__(self, item: tuple) -> tuple:
        img, target = item
        img = self.t(img)
        return img, target


class Brightness(object):
    def __init__(self, p=0.5, brightness=2):
        self.p = p
        self.b = T.ColorJitter(brightness=brightness)

    def __call__(self, item: tuple) -> tuple:
        img, target = item
        if random.random() <= self.p:
            img = self.b(img)
        return img, target


class Contrast(object):
    def __init__(self, p=0.5, contrast=2):
        self.p = p
        self.b = T.ColorJitter(contrast=contrast)

    def __call__(self, item: tuple) -> tuple:
        img, target = item
        if random.random() <= self.p:
            img = self.b(img)
        return img, target


class Hue(object):
    def __init__(self, p=0.5, hue=0.25):
        self.p = p
        self.b = T.ColorJitter(hue=hue)

    def __call__(self, item: tuple) -> tuple:
        img, target = item
        if random.random() <= self.p:
            img = self.b(img)
        return img, target


class Saturation(object):
    def __init__(self, p=0.5, saturation=2):
        self.p = p
        self.b = T.ColorJitter(saturation=saturation)

    def __call__(self, item: tuple) -> tuple:
        img, target = item
        if random.random() <= self.p:
            img = self.b(img)
        return img, target


class RandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p
        self.f = T.RandomHorizontalFlip(p=1.0)

    def __call__(self, item: tuple) -> tuple:
        img, target = item
        if random.random() <= self.p:
            img = self.f(img)

            gates = target['gates'] if 'gates' in target else target['boxes']
            new_gates = []
            for gate in gates:
                new_gates.append([
                    1.0 - gate[6],
                    gate[7],
                    1.0 - gate[4],
                    gate[5],
                    1.0 - gate[2],
                    gate[3],
                    1.0 - gate[0],
                    gate[1]
                ])
            gates = torch.tensor(new_gates, dtype=torch.float32)

            if 'gates' in target:
                _, w = target['size']
                boxes = target['boxes']
                n_boxes = []
                for box in boxes:
                    n_boxes.append([
                        w - box[2],
                        box[1],
                        w - box[0],
                        box[3]
                    ])
                boxes = torch.tensor(n_boxes, dtype=torch.float32)
            upd_dict = {}

            if 'gates' in target:
                upd_dict['boxes'] = boxes
                upd_dict['gates'] = gates
            else:
                upd_dict['boxes'] = gates
            target.update(upd_dict)

        return img, target


'''
    target = {
                'boxes': torch.tensor(bnd_boxes, dtype=torch.float32),
                'gates': torch.tensor(gates, dtype=torch.float32),
                'labels': torch.tensor([1 for _ in range(len(gates))], dtype=torch.int64),
                'image_id': torch.tensor([len(gates)], dtype=torch.int64),
                'area': torch.tensor(areas, dtype=torch.float32),
                'iscrowd': torch.tensor([0 for _ in range(len(gates))], dtype=torch.int64),
                'orig_size': torch.tensor(orig_shape, dtype=torch.int64),
                'size': torch.tensor(orig_shape, dtype=torch.int64)
            }
'''


class Resize(object):
    def __init__(self, dim):
        if isinstance(dim, int):
            self.r = T.Resize([dim, dim])
        elif isinstance(dim, (tuple, list)):
            self.r = T.Resize(dim)
        else:
            raise Exception("Expected int, tuple or list as input")

    def __call__(self, item: tuple) -> tuple:
        img, target = item

        img = self.r(img)
        upd = {
            'size': torch.tensor([img.shape[1], img.shape[2]], dtype=torch.int64)
        }

        if 'gates' in target:
            height, width = target['size']
            n_height, n_width = img.shape[1], img.shape[2]
            boxes = target['boxes']
            n_boxes = []
            for box in boxes:
                x0 = box[0] / width * n_width
                y0 = box[1] / height * n_height
                x1 = box[2] / width * n_width
                y1 = box[3] / height * n_height
                n_boxes.append([x0, y0, x1, y1])
            upd['boxes'] = torch.tensor(n_boxes, dtype=torch.float32)

        target.update(upd)

        return img, target


def get_masks(target: dict) -> torch.Tensor:
    gates = target['gates'] if 'gates' in target else target['boxes']
    height, width = target['size']

    masks = []
    for gate in gates:  # gate has dim [8]
        a = (gate[0] * width, gate[1] * height)
        b = (gate[2] * width, gate[3] * height)
        c = (gate[4] * width, gate[5] * height)
        d = (gate[6] * width, gate[7] * height)

        img = Image.new('L', (width, height), 0)
        ImageDraw.Draw(img).polygon([a, b, c, d], outline=1, fill=1)
        mask = torch.tensor(np.array(img), dtype=torch.int8)
        masks.append(mask)

    return torch.stack(masks)


def fix_bbox(bbox: list) -> list:
    min_x = min([bbox[0], bbox[2]])
    max_x = max([bbox[0], bbox[2]])
    min_y = min([bbox[1], bbox[3]])
    max_y = max([bbox[1], bbox[3]])

    if abs(min_x - max_x) < 0.1:
        max_x += 3

    if abs(min_y - max_y) < 0.1:
        max_y += 3

    return [min_x, min_y, max_x, max_y]


class RealGatesDS(torch.utils.data.Dataset):

    img_extension = '.jpg'

    folders = {
        # "basement_course1": ".xml",
        # "basement_course3": ".xml",
        # "bebop_merge": ".xml",
        # "bebop_merge_distort": ".xml",
        # "cyberzoo": ".xml",
        "daylight15k": ".xml",
        "daylight_course1": ".xml",
        "daylight_course3": ".xml",
        "daylight_course5": ".xml",
        "daylight_flight": ".xml",
        # "eth": ".pkl",
        # "google_merge_distort": ".xml",
        "iros2018_course1": ".xml",
        "iros2018_course3_test": ".xml",
        "iros2018_course5": ".xml",
        "iros2018_flights": ".xml",
        "iros2018_frontal": ".xml",
        # "random_flight": ".xml"
    }

    # orders = {
    #     "basement_course1": True,
    #     "basement_course3": False,
    #     "bebop_merge": None,
    #     "bebop_merge_distort": None,
    #     "cyberzoo": True,
    #     "daylight15k": True,
    #     "daylight_course1": True,
    #     "daylight_course3": True,
    #     "daylight_course5": False,
    #     "daylight_flight": None,
    #     "eth": None,
    #     "google_merge_distort": None,
    #     "iros2018_course1": True,
    #     "iros2018_course3_test": True,
    #     "iros2018_course5": True,
    #     "iros2018_flights": True,
    #     "iros2018_frontal": True,
    #     "random_flight": True
    # }

    std_transform = T.Compose([
        ToTensor(),
        Resize((256, 256)),
        RandomFlip(p=0.5)
        # Brightness(p=0.1, brightness=2),
        # Contrast(p=0.1, contrast=2),
        # Saturation(p=0.1, saturation=2),
        # Hue(p=0.1, hue=0.25)
    ])

    def __init__(self, path, image_set, transform=None, backup_list_path='', mask_rcnn=False):

        assert 'train' in image_set or 'val' in image_set

        print("[RG DATASET] INITIALIZING Real Gates dataset..")
        if mask_rcnn:
            print("[RG DATASET] Initializing for Mask-RCNN")
        self.mask_rcnn = mask_rcnn

        path_dirs = listdir(path)

        for folder in self.folders:
            assert folder in path_dirs, f"'{folder}' directory is missing from path"

        if transform is None:
            self.transform = self.std_transform
        else:
            self.transform = transform
        self.files = []

        if 'val' in image_set:
            print("[RG DATASET] Returning empty dataloader for 'val' set")
            return

        if backup_list_path != '' and isfile(backup_list_path):
            print("[RG DATASET] Loading list from backup at", backup_list_path)
            with open(backup_list_path, 'rb') as file:
                self.files = pickle.load(file)
                print(f"[RG DATASET] Reloaded dataset containing {len(self.files)} images from backup.")
                return

        min_gates = 999
        max_gates = -1

        for folder in self.folders:
            print("[RG DATASET] Creating index for folder:", folder)
            folder_path = join(path, folder)
            all_files = listdir(folder_path)
            print(f"[RG DATASET] containing {len(all_files)} files")
            for file in all_files:
                if isfile(join(folder_path, file)) and 'jpg' in file:
                    xml_path = join(folder_path, file).replace('.jpg', self.folders[folder])
                    num_gates = len(ET.parse(xml_path).getroot().findall('object'))
                    if num_gates > 0:
                        min_gates = num_gates if num_gates < min_gates else min_gates
                        max_gates = num_gates if num_gates > max_gates else max_gates
                        img = Image.open(file)
                        width, height = img.size
                        shape = (height, width)
                        self.files.append((xml_path, shape))

        if backup_list_path != '':
            print(f"[RG DATASET] Saving a copy of the list at {backup_list_path}")
            with open(backup_list_path, 'wb') as file:
                pickle.dump(self.files, file)
            print("[RG DATASET] Saved list as backup")

        print(f"[RG DATASET] Created dataset containing {len(self.files)} images. Images contain {min_gates} to {max_gates} gates")

    def __len__(self):
        return len(self.files)

    def get_height_and_width(self, index):
        return self.files[index][1]

    def __getitem__(self, item):
        """XML CONVENTION
            annotation
            filename
            object
                name
                bndbox --> xmin, ymin, xmax, ymax
                pose --> north, east, down, yaw, pitch, roll
                gate_corners --> top_left, top_right, bottom_right, bottom_left, center
            object..
        """

        item_path = self.files[item][0]

        # Open the image as PIL
        img = Image.open(item_path.replace('.xml', '.jpg'))

        width, height = img.size
        orig_shape = [height, width]

        xml = ET.parse(item_path).getroot()

        gates = []
        if self.mask_rcnn:
            bnd_boxes = []

        xml_objects = xml.findall('object')
        assert xml_objects is not None, "No gates found in XML file"

        for obj in xml_objects:
            assert obj.find('name').text == 'gate', "XML gate object not named 'gate'"
            assert obj.find('gate_corners') is not None

            gate_corners_dict = {}

            for corner in obj.find('gate_corners')[:4]:
                x = float(corner.text.split(',')[0]) / width
                y = (height - float(corner.text.split(',')[1])) / height
                x_y = [x, y]
                gate_corners_dict[corner.tag] = x_y

            # This is done to follow the usual convention of starting with the BL corner and go clockwise
            gate_corners = [
                gate_corners_dict['bottom_left'][0],
                gate_corners_dict['bottom_left'][1],
                gate_corners_dict['top_left'][0],
                gate_corners_dict['top_left'][1],
                gate_corners_dict['top_right'][0],
                gate_corners_dict['top_right'][1],
                gate_corners_dict['bottom_right'][0],
                gate_corners_dict['bottom_right'][1]
            ]
            gates.append(gate_corners)

            if self.mask_rcnn:
                corners = obj.find('bndbox')
                assert corners is not None
                box_corners = [float(corner.text) for corner in corners]
                box_corners[1] = height - box_corners[1]
                box_corners[3] = height - box_corners[3]
                box_corners = fix_bbox(box_corners)
                bnd_boxes.append(box_corners)

        areas = []
        for gate in gates:
            poly_gate = [
                [gate[0], gate[1]],
                [gate[2], gate[3]],
                [gate[4], gate[5]],
                [gate[6], gate[7]]
            ]
            poly = Polygon(poly_gate)
            areas.append(poly.area)

        if self.mask_rcnn:
            target = {
                'boxes': torch.tensor(bnd_boxes, dtype=torch.float32),
                'gates': torch.tensor(gates, dtype=torch.float32),
                'labels': torch.tensor([1 for _ in range(len(gates))], dtype=torch.int64),
                'image_id': torch.tensor([len(gates)], dtype=torch.int64),
                'area': torch.tensor(areas, dtype=torch.float32),
                'iscrowd': torch.tensor([0 for _ in range(len(gates))], dtype=torch.int64),
                'orig_size': torch.tensor(orig_shape, dtype=torch.int64),
                'size': torch.tensor(orig_shape, dtype=torch.int64)
            }
        else:
            target = {
                'boxes': torch.tensor(gates, dtype=torch.float32),
                'labels': torch.tensor([0 for _ in range(len(gates))], dtype=torch.int64),
                'image_id': torch.tensor([len(gates)], dtype=torch.int64),
                'area': torch.tensor(areas, dtype=torch.float32),
                'iscrowd': torch.tensor([0 for _ in range(len(gates))], dtype=torch.int64),
                'orig_size': torch.tensor(orig_shape, dtype=torch.int64),
                'size': torch.tensor(orig_shape, dtype=torch.int64)
            }

        ###############################
        # TRANSFORMS
        ###############################

        if self.transform:
            img, target = self.transform((img, target))

        ###############################

        mask_dict = {}
        masks = get_masks(target)
        if self.mask_rcnn:
            mask_dict['boxes'] = torch.clamp(target['boxes'], min=0.0)
            mask_dict['gates'] = torch.clamp(target['gates'], min=0.0, max=1.0)
        else:
            mask_dict['boxes'] = torch.clamp(target['boxes'], min=0.0, max=1.0)

        mask_dict['masks'] = masks

        target.update(mask_dict)

        return img, target


if __name__ == '__main__':

    ds = RealGatesDS(
        "/home/andreaalf/Documents/thesis/datasets/gate_full_sample",
        image_set='train',
        mask_rcnn=False
    )

    for i, (img, target) in enumerate(ds):
        plt.imshow(img.cpu().permute(1, 2, 0))
        h, w = img.shape[-2:]
        bnd_box = target['boxes']
        for box in bnd_box:
            plt.scatter([box[2]*w, box[4]*w, box[6]*w], [box[3]*h, box[5]*h, box[7]*h])
            plt.scatter([box[0]*w], [box[1]*h])
        plt.show()

        # for mask in target['masks']:
        #     plt.imshow(mask)
        #     plt.show()
        # h, w = img.shape[-2:]
        #
        # boxes = target['boxes']
        # for gate in boxes:
        #     x = torch.tensor([
        #         gate[0],
        #         gate[2],
        #         gate[4],
        #         gate[6]
        #     ])
        #     y = torch.tensor([
        #         gate[1],
        #         gate[3],
        #         gate[5],
        #         gate[7]
        #     ])
        #     plt.scatter(x*w, y*h)
        #
        # plt.show()
