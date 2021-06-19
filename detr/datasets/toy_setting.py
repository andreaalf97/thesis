import numpy as np
import cv2
import random
import torch

import torchvision.transforms as T
import matplotlib.pyplot as plt


COLOR = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255)
}

class PolyGate:

    def __init__(self, image_height, image_width, color='none', stroke=-1, num_points=20):

        self.image_height, self.image_width = image_height, image_width

        x = list(np.linspace(0.10, 0.90, num_points))
        noise = [random.random() * 0.3 - 0.15 for _ in x]
        y = [(1 - i) + (n) for i, n in zip(x, noise)]
        x = [i + n for i, n in zip(x, noise)]

        x = [max(0, min(1, i)) for i in x]
        y = [max(0, min(1, i)) for i in y]

        self.points = [(round(i * image_width), round(j * image_height)) for i, j in zip(x, y)]

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
        return [[float(x) / self.image_width, float(y) / self.image_height] for x, y in self.points]

    def get_area(self) -> float:
        return 1

    def get_class(self) -> int:
        return 0


def get_ts_image(height, width, num_gates=3, black_and_white=True, stroke=-1) -> (np.ndarray, list, list):

    img = get_ts_background(height, width, bgr=False, black_white=black_and_white)

    if black_and_white:
        gate = PolyGate(height, width, color='white', stroke=stroke, num_points=num_gates)
    else:
        gate = PolyGate(height, width, color='none', stroke=stroke, num_points=num_gates)

    labels = gate.get_labels()
    areas = gate.get_area()
    classes = gate.get_class()
    img = print_polygate(img, gate)

    return img, labels, areas, classes


def print_polygate(img: np.ndarray, gate: PolyGate) -> np.ndarray:

    for i in range(1, len(gate.points)):
        img = cv2.line(
            img,
            (gate.points[i-1][0], gate.points[i-1][1]),
            (gate.points[i][0], gate.points[i][1]),
            color=gate.color,
            thickness=2,
            lineType=cv2.LINE_AA
        )

    img = cv2.circle(
        img,
        (gate.points[0][0], gate.points[0][1]),
        radius=4,
        color=COLOR['green'],
        thickness=-1,
        lineType=cv2.LINE_AA
    )
    for x, y in gate.points[1:]:
        img = cv2.circle(
            img,
            (x, y),
            radius=4,
            color=COLOR['green'],
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

    def __init__(self, img_height, img_width, num_gates=3, black_and_white=True, stroke=-1):
        self.img_height = img_height
        self.img_width = img_width
        self.num_gates = num_gates
        self.black_and_white = black_and_white
        self.stroke = stroke
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
            black_and_white=self.black_and_white,
            stroke=self.stroke
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
        num_gates=8,
        black_and_white=True,
        stroke=2
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
