import torch
import torchvision.transforms as T
from os import listdir
from os.path import join, isfile
import xml.etree.ElementTree as ET
from PIL import Image
from shapely.geometry import Polygon
import matplotlib.pyplot as plt


class ToTensor(object):
    def __init__(self):
        self.t = T.ToTensor()

    def __call__(self, item: tuple) -> tuple:
        img, target = item
        img = self.t(img)
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
        "basement_course1": ".xml",
        "basement_course3": ".xml",
        "bebop_merge": ".xml",
        "bebop_merge_distort": ".xml",
        "cyberzoo": ".xml",
        "daylight15k": ".xml",
        "daylight_course1": ".xml",
        "daylight_course3": ".xml",
        "daylight_course5": ".xml",
        "daylight_flight": ".xml",
        "eth": ".pkl",
        "google_merge_distort": ".xml",
        "iros2018_course1": ".xml",
        "iros2018_course3_test": ".xml",
        "iros2018_course5": ".xml",
        "iros2018_flights": ".xml",
        "iros2018_frontal": ".xml",
        "random_flight": ".xml"
    }

    def __init__(self, path, image_set, transform=None):

        assert 'train' in image_set or 'val' in image_set
        path_dirs = listdir(path)

        for folder in self.folders:
            assert folder in path_dirs, f"'{folder}' directory is missing from path"

        self.transform = transform
        self.files = []

        for folder in self.folders:
            folder_path = join(path, folder)
            all_files = listdir(folder_path)
            for file in all_files:
                if isfile(join(folder_path, file)) and 'jpg' in file:
                    self.files.append(join(folder_path, file).replace('.jpg', self.folders[folder]))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):

        print(self.files[item])

        img = Image.open(self.files[item].split('.')[0] + '.jpg')

        tens = T.ToTensor()
        orig_shape = [tens(img).shape[1], tens(img).shape[2]]
        height, width = orig_shape[0], orig_shape[1]

        xml = ET.parse(self.files[item]).getroot()

        gates = []
        for obj in xml.findall('object'):

            # First we check if the object is a GATE object
            not_gate = False
            for name in obj.find('name'):
                if name.text != 'gate':
                    not_gate = True
            if not_gate:
                print("NOT A GATE @", self.files[item])
                continue

            gate_corners = []

            for corner in obj.find('gate_corners'):
                gate_corners.append([
                    float(corner.text.split(',')[0]) / width,
                    float(corner.text.split(',')[1]) / height
                ])

            # This is done to follow the usual convention of starting with the BL corner and go clockwise
            gate_corners = [
                gate_corners[3],
                gate_corners[0],
                gate_corners[1],
                gate_corners[2]
            ]
            gates.append(gate_corners)

        areas = []
        for gate in gates:
            poly = Polygon(gate)
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

    transform = T.Compose([
        ToTensor()
        # Resize([256, 256])
    ])

    ds = RealGatesDS(
        "/home/andreaalf/Documents/thesis/datasets/gate_full",
        image_set='train',
        transform=transform
    )

    for img, target in ds:
        plt.imshow(img.cpu().permute(1, 2, 0))

        h, w = img.shape[-2:]

        gates = target["boxes"]
        for gate in gates:
            plt.scatter(gate[:, 0] * w, gate[:, 1] * h)

        # plt.show()
        # break
        # shape = str(list(img.shape))
        #
        # if shape not in shapes:
        #     shapes[shape] = 1
        # else:
        #     shapes[shape] += 1




    exit(0)