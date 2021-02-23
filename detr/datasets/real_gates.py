import torch
import torchvision.transforms as T
from os import listdir
from os.path import join, isfile
import xml.etree.ElementTree as ET
from PIL import Image
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import random


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
            boxes = torch.clone(target['boxes'])
            for gate in boxes:
                gate[0] = 1.0 - gate[0]
                gate[2] = 1.0 - gate[2]
                gate[4] = 1.0 - gate[4]
                gate[6] = 1.0 - gate[6]
            upd_dict = {
                'boxes': boxes
            }
            target.update(upd_dict)

        return img, target


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
        target.update(upd)

        return img, target


class RealGatesDS(torch.utils.data.Dataset):
    img_extension = '.jpg'

    folders = {
        # "basement_course1": ".xml",
        # "basement_course3": ".xml",
        # "bebop_merge": ".xml",
        # "bebop_merge_distort": ".xml",
        # "cyberzoo": ".xml",
        "daylight15k": ".xml",
        # "daylight_course1": ".xml",
        # "daylight_course3": ".xml",
        # "daylight_course5": ".xml",
        # "daylight_flight": ".xml",
        # "eth": ".pkl",
        # "google_merge_distort": ".xml",
        # "iros2018_course1": ".xml",
        # "iros2018_course3_test": ".xml",
        # "iros2018_course5": ".xml",
        # "iros2018_flights": ".xml",
        # "iros2018_frontal": ".xml",
        # "random_flight": ".xml"
    }

    std_transform = T.Compose([
        ToTensor(),
        Resize([256, 256]),
        RandomFlip(p=0.5),
        Brightness(p=0.1, brightness=2),
        Contrast(p=0.1, contrast=2),
        Saturation(p=0.1, saturation=2),
        Hue(p=0.1, hue=0.25)
    ])

    def __init__(self, path, image_set, transform=None):

        assert 'train' in image_set or 'val' in image_set

        print("[RG DATASET] INITIALIZING Real Gates dataset..")

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

        for folder in self.folders:
            print("[RG DATASET] Creating index for folder:", folder)
            folder_path = join(path, folder)
            all_files = listdir(folder_path)
            print(f"[RG DATASET] containing {len(all_files)} files")
            for file in all_files:
                if isfile(join(folder_path, file)) and 'jpg' in file:
                    xml_path = join(folder_path, file).replace('.jpg', self.folders[folder])
                    if len(ET.parse(xml_path).getroot().findall('object')) > 0:
                        self.files.append(xml_path)

        print(f"[RG DATASET] Created dataset containing {len(self.files)} images.")

    def __len__(self):
        return len(self.files)

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

        item_path = self.files[item]

        img = Image.open(item_path.replace('.xml', '.jpg'))

        tens = T.ToTensor()
        orig_shape = [tens(img).shape[1], tens(img).shape[2]]
        height, width = orig_shape[0], orig_shape[1]

        xml = ET.parse(item_path).getroot()

        gates = []
        xml_objects = xml.findall('object')
        assert xml_objects is not None, "No gates found in XML file"

        for obj in xml_objects:
            assert obj.find('name').text == 'gate', "XML gate object not named 'gate"
            assert obj.find('gate_corners') is not None

            gate_corners = []

            for corner in obj.find('gate_corners'):
                gate_corners.append(
                    float(corner.text.split(',')[0]) / width
                )
                gate_corners.append(
                    (height-float(corner.text.split(',')[1])) / height
                )
            # This is done to follow the usual convention of starting with the BL corner and go clockwise
            gate_corners = [
                gate_corners[6],
                gate_corners[7],
                gate_corners[0],
                gate_corners[1],
                gate_corners[2],
                gate_corners[3],
                gate_corners[4],
                gate_corners[5],

            ]
            gates.append(gate_corners)

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

        return img, target


if __name__ == '__main__':

    ds = RealGatesDS(
        "/home/andreaalf/Documents/thesis/datasets/gate_full_sample",
        image_set='train'
    )

    for img, target in ds:
        plt.imshow(img.cpu().permute(1, 2, 0))

        h, w = img.shape[-2:]

        boxes = target['boxes']
        for gate in boxes:
            x = torch.tensor([
                gate[0],
                gate[2],
                gate[4],
                gate[6]
            ])
            y = torch.tensor([
                gate[1],
                gate[3],
                gate[5],
                gate[7]
            ])
            plt.scatter(x*w, y*h)

        plt.show()
