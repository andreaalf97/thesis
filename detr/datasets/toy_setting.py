import numpy as np
import cv2
import random
import torch
import math

from PIL import Image, ImageDraw
import torchvision.transforms as T
import matplotlib.pyplot as plt


class PolyGate:

    def __init__(self, image_height, image_width, color='none', stroke=-1):

        self.image_height, self.image_width = image_height, image_width

        self.x = random.randint(int(0.05*image_width), int(image_width - (0.05*image_width)))
        self.y = random.randint(int(0.05*image_height), int(image_height - (0.05*image_height)))

        if stroke == -1:
            self.stroke = random.randint(1, 3)
        else:
            self.stroke = stroke

        if 'white' in color.lower():
            self.color = (255, 255, 255)
        elif 'black' in color.lower():
            self.color = (0, 0, 0)
        elif 'none' in color.lower():
            self.color = (
                int(random.random() * 255),
                int(random.random() * 255),
                int(random.random() * 255)
            )
        else:
            raise Exception(f"Color {color} not supported")

    def get_labels(self) -> list:
        return [float(self.x) / self.image_width, float(self.y) / self.image_height]

    def get_area(self) -> float:
        return 1

    def get_class(self) -> int:
        return 0


def get_ts_image(height, width, num_gates=3, no_gate_chance=0.10, black_and_white=True, stroke=-1, fix_gates=False) -> (np.ndarray, list, list):

    if random.random() < no_gate_chance:
        num_gates = 0
    else:
        if not fix_gates:
            num_gates = random.randint(1, num_gates)

    img = get_ts_background(height, width, bgr=False, black_white=black_and_white)

    labels = []
    areas = []
    classes = []

    for _ in range(num_gates):
        if black_and_white:
            gate = PolyGate(height, width, color='white', stroke=stroke)
        else:
            gate = PolyGate(height, width, color='none', stroke=stroke)

        labels.append(gate.get_labels())
        areas.append(gate.get_area())
        classes.append(gate.get_class())
        img = print_polygate(img, gate)

    return img, labels, areas, classes


def print_polygate(img: np.ndarray, gate: PolyGate) -> np.ndarray:

    img = cv2.circle(
        img,
        (gate.x, gate.y),
        radius=gate.stroke,
        color=gate.color,
        thickness=-1,
        lineType=cv2.LINE_AA
    )
    return img


def get_ts_background(height, width, bgr=False, black_white=False) -> np.ndarray:

    if black_white:
        image = np.zeros(
            (height, width, 3),
            dtype=np.uint8
        )
        image[:, :] = [0, 0, 0]
        return image

    red = random.randint(0, 255)
    green = random.randint(0, 255)
    blue = random.randint(0, 255)

    image = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    if bgr:
        image[:, :] = [blue, green, red]
    else:
        image[:, :] = [red, green, blue]

    return image


def display_image(image: np.ndarray, labels):
    cv2.imshow("THERE ARE {} GATES".format(len(labels)), image)
    print(labels)
    cv2.waitKey()
    cv2.destroyAllWindows()


class ToTensor(object):
    def __init__(self):
        self.t = T.ToTensor()

    def __call__(self, sample):
        img, target = sample
        img = self.t(img)
        return img, target


class TSDataset(torch.utils.data.Dataset):

    def __init__(self, img_height, img_width, num_gates=3, black_and_white=True,
                 no_gate_chance=0.0, stroke=-1, fix_gates=False):
        self.img_height = img_height
        self.img_width = img_width
        self.num_gates = num_gates
        self.black_and_white = black_and_white
        self.no_gate_chance = no_gate_chance
        self.stroke = stroke
        self.fix_gates = fix_gates
        self.transform = T.Compose([
                ToTensor(),
                # Clamp(),
            ])
        self.label = 0

    def __len__(self):
        return 10000

    def __getitem__(self, index):
        image, labels, areas, classes = get_ts_image(
            self.img_height,
            self.img_width,
            num_gates=self.num_gates,
            no_gate_chance=self.no_gate_chance,
            black_and_white=self.black_and_white,
            stroke=self.stroke,
            fix_gates=self.fix_gates
        )

        # boxes, labels, image_id, area, iscrowd, orig_size, size
        target = {
            'boxes': torch.tensor(labels, dtype=torch.float32),
            'labels': torch.tensor([self.label for _ in range(len(labels))], dtype=torch.int64),
            'image_id': torch.tensor([len(labels)], dtype=torch.int64),
            'area': torch.tensor(areas, dtype=torch.float32),
            'iscrowd': torch.tensor([0 for _ in range(len(labels))], dtype=torch.int64),
            'orig_size': torch.tensor([self.img_height, self.img_width], dtype=torch.int64),
            'size': torch.tensor([self.img_height, self.img_width], dtype=torch.int64),
        }

        if self.transform:
            image, target = self.transform((image, target))

        return image, target


if __name__ == '__main__':

    ds = TSDataset(
        256, 256,
        num_gates=20,
        black_and_white=True,
        no_gate_chance=0.0,
        stroke=2,
    )

    for image, target in ds:
        plt.imshow(image.cpu().permute(1, 2, 0))
        plt.show()

        plt.imshow(image.cpu().permute(1, 2, 0))
        for point in target['boxes']:
            plt.scatter(point[0] * 256, point[1] * 256)
        plt.show()
        print(target)
        break
