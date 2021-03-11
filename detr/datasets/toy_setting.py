import numpy as np
import cv2
import random
import torch
import math

from PIL import Image, ImageDraw
import torchvision.transforms as T
import matplotlib.pyplot as plt


class PolyGate:
    def __init__(self, image_height, image_width, color='none', num_corners=-1, stroke=-1):

        self.image_height, self.image_width = image_height, image_width

        self.num_corners = num_corners

        self.c_x = random.randint(int(0.20*image_width), int(image_width - (0.20*image_width)))
        self.c_y = random.randint(int(0.20*image_height), int(image_height - (0.20*image_height)))

        radius_perc = random.uniform(0.05, 0.50)
        self.radius = image_height * radius_perc

        if num_corners == -1:
            num_corners = random.randint(3, 8)

        self.corners = []

        alpha_zero = (2 * math.pi / num_corners)
        alpha = random.uniform(math.pi/2, math.pi)
        for _ in range(num_corners):
            x = self.c_x + self.radius * math.cos(alpha)
            y = self.c_y + self.radius * math.sin(alpha)
            slope = math.tan(alpha) if alpha != math.pi/2 else math.tan(alpha + 0.001)
            delta_x = random.random() * 0.10 * self.radius - 0.05
            delta_y = slope * delta_x
            self.corners.append((int(x+delta_x), int(y+delta_y)))
            alpha += alpha_zero

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
        labels = []

        for corner in self.corners:
            labels.append(corner[0]/self.image_width)
            labels.append(corner[1]/self.image_height)

        return labels

    def get_area(self) -> float:
        return math.pi * (self.radius**2)

    def get_class(self) -> int:
        return len(self.corners)


def get_ts_image(height, width, num_gates=3, no_gate_chance=0.10, black_and_white=True, stroke=-1, num_corners=-1) -> (np.ndarray, list, list):

    if num_gates == -1:
        if random.random() < no_gate_chance:
            num_gates = 0
        else:
            num_gates = random.randint(1, num_gates)

    img = get_ts_background(height, width, bgr=False, black_white=black_and_white)

    labels = []
    areas = []
    classes = []

    for _ in range(num_gates):
        if black_and_white:
            gate = PolyGate(height, width, color='white', stroke=stroke, num_corners=num_corners)
        else:
            gate = PolyGate(height, width, color='none', stroke=stroke, num_corners=num_corners)

        labels.append(gate.get_labels())
        areas.append(gate.get_area())
        classes.append(gate.get_class())
        img = print_polygate(img, gate)

    return img, labels, areas, classes


def print_polygate(img: np.ndarray, gate: PolyGate) -> np.ndarray:

    for i in range(1, len(gate.corners)):
        img = cv2.line(
            img,
            gate.corners[i-1],
            gate.corners[i],
            gate.color,
            gate.stroke,
            lineType=cv2.LINE_AA
        )

    img = cv2.line(
        img,
        gate.corners[-1],
        gate.corners[0],
        gate.color,
        gate.stroke,
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


class Clamp(object):
    def __call__(self, sample):
        img, target = sample
        boxes = target['boxes']
        if 'masks' not in target:
            target['boxes'] = torch.clamp(boxes, 0.0, 1.0)
        else:
            h, w = target['size']
            gates = []
            for gate in target['boxes']:
                gates.append([
                    torch.clamp(gate[0], 0, w).item(),
                    torch.clamp(gate[1], 0, h).item(),
                    torch.clamp(gate[2], 0, w).item(),
                    torch.clamp(gate[3], 0, h).item(),
                    torch.clamp(gate[4], 0, w).item(),
                    torch.clamp(gate[5], 0, h).item(),
                    torch.clamp(gate[6], 0, w).item(),
                    torch.clamp(gate[7], 0, h).item()
                ])
            target['boxes'] = torch.tensor(gates, dtype=torch.float32)
        return img, target


class MaskRCNN(object):
    def __call__(self, sample):
        img, target = sample

        h, w = target['size']

        gates = []
        for gate in target['boxes']:
            gates.append([
                gate[0].item() * w,
                gate[1].item() * h,
                gate[2].item() * w,
                gate[3].item() * h,
                gate[4].item() * w,
                gate[5].item() * h,
                gate[6].item() * w,
                gate[7].item() * h
            ])

        target['boxes'] = torch.tensor(gates, dtype=torch.float32)

        masks = []
        for label in target['boxes']:
            mask_img = Image.new('L', (w, h), 0)
            ImageDraw.Draw(mask_img).polygon([
                (label[0], label[1]),
                (label[2], label[3]),
                (label[4], label[5]),
                (label[6], label[7]),
            ], outline=1, fill=1)
            mask_img = torch.tensor(np.array(mask_img), dtype=torch.int8)
            masks.append(mask_img)
        target['masks'] = torch.stack(masks)

        return img, target


class TSDataset(torch.utils.data.Dataset):

    std_transform = T.Compose([
        ToTensor(),
        # MaskRCNN(),
        Clamp()
    ])

    def __init__(self, img_height, img_width, num_gates=3, black_and_white=True,
                 no_gate_chance=0.0, stroke=-1, num_corners=4):
        self.img_height = img_height
        self.img_width = img_width
        self.num_gates = num_gates
        self.black_and_white = black_and_white
        self.no_gate_chance = no_gate_chance
        self.stroke = stroke
        self.num_corners = num_corners
        self.transform = self.std_transform

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
            num_corners=self.num_corners
        )

        # boxes, labels, image_id, area, iscrowd, orig_size, size
        target = {
            'boxes': torch.tensor(labels, dtype=torch.float32),
            'labels': torch.tensor(classes, dtype=torch.int64),
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

    mask_rcnn = False
    h, w = 256, 256
    ds = TSDataset(h, w)

    for image, target in ds:
        plt.imshow(image.cpu().permute(1, 2, 0))

        for gate in target['boxes']:
            if mask_rcnn:
                plt.scatter([gate[2], gate[4], gate[6]], [gate[3], gate[5], gate[7]])
                plt.scatter([gate[0]], [gate[1]])
            else:
                plt.scatter([gate[2]*w, gate[4]*w, gate[6]*w], [gate[3]*h, gate[5]*h, gate[7]*h])
                plt.scatter([gate[0]*w], [gate[1]*h])

        print('\n'.join([str(k) + ' --> ' + str(target[k]) for k in target]))

        plt.show()

        if 'masks' in target:
            for mask in target['masks']:
                plt.imshow(mask)
                plt.show()
        break
