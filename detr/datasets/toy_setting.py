import numpy as np
import cv2
import random
import torch
import math

from PIL import Image, ImageDraw
import torchvision.transforms as T
import matplotlib.pyplot as plt


CLASSES = {
    "<start>": 0,
    "<point>": 1,
    "<end-of-computation>": 2
}

class PolyGate:

    MIN_CORNERS = 3
    MAX_CORNERS = 7

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


def validate_poly(corners):

    if len(corners) == 3:
        return True

    lines = []

    # Create list of lines
    lines.append([corners[0], corners[1]])
    for i in range(1, len(corners)-1):
        lines.append([corners[i], corners[i+1]])
    lines.append([corners[-1], corners[0]])

    for i in range(len(lines)):
        temp_lines = lines.copy()
        temp_lines.remove(lines[i + 1] if i + 1 < len(lines) else lines[0])
        temp_lines.remove(lines[i])
        temp_lines.remove(lines[i - 1])

        for line_2 in temp_lines:
            if line_intersection(lines[i], line_2):
                return False

    return True


def line_intersection(line1, line2):

    a, b = line1
    c, d = line2

    if clockwise(a, b, c) * clockwise(a, b, d) > 0:
        return False
    if clockwise(c, d, a) * clockwise(c, d, b) > 0:
        return False
    return True


def clockwise(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c

    return (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)


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


class GetSentence(object):
    def __init__(self, seq_order='lr'):
        self.seq_order = seq_order

    def __call__(self, sample):
        img, target = sample

        sequence = []
        start_token = torch.zeros(256)
        start_token[2 + CLASSES['<start>']] = 1
        sequence.append(start_token)

        if self.seq_order == 'random':
            raise Exception("Not ready for random ordering of polygons")
        elif self.seq_order in ['lr', 'rl', 'tb', 'bt']:
            centers = []
            for polygon in target['boxes']:
                mean_x = polygon[0].item()
                mean_y = polygon[1].item()
                centers.append((mean_x, mean_y))
            centers = {i: c for i, c in enumerate(centers)}

            if self.seq_order == 'lr':
                centers = sorted(centers.items(), key=lambda x: x[1][0])
            elif self.seq_order == 'rl':
                centers = sorted(centers.items(), key=lambda x: x[1][0], reverse=True)
            elif self.seq_order == 'tb':
                centers = sorted(centers.items(), key=lambda x: x[1][1])
            elif self.seq_order == 'bt':
                centers = sorted(centers.items(), key=lambda x: x[1][1], reverse=True)

            for i, center in centers:
                polygon = target['boxes'][i]
                token = torch.zeros(256)
                token[2 + CLASSES['<point>']] = 1
                token[:2] = polygon
                sequence.append(token)
        elif self.seq_order in ['sl', 'ls']:
            raise Exception("No ordering based on area for points")

        end_computation = torch.zeros(256)
        end_computation[2 + CLASSES['<end-of-computation>']] = 1
        sequence.append(end_computation)

        sequence = torch.stack(sequence)

        target['sequence'] = sequence
        return img, target

class TSDataset(torch.utils.data.Dataset):

    def __init__(self, img_height, img_width, num_gates=3, black_and_white=True,
                 no_gate_chance=0.0, stroke=-1, seq_order='tb', fix_gates=False):
        assert seq_order in ('ls', 'sl', 'lr', 'rl', 'tb', 'bt', 'random'), f"{seq_order} order not implemented for TOY SETTING"
        self.img_height = img_height
        self.img_width = img_width
        self.num_gates = num_gates
        self.black_and_white = black_and_white
        self.no_gate_chance = no_gate_chance
        self.stroke = stroke
        self.seq_order = seq_order
        self.fix_gates = fix_gates
        self.transform = T.Compose([
                ToTensor(),
                # Clamp(),
                GetSentence(seq_order=self.seq_order)
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

    ds = TSDataset(256, 256, num_gates=4, black_and_white=True, no_gate_chance=0.0, stroke=-1, seq_order='lr')

    start = 0
    point = 0
    end_poly = 0
    end = 0

    tot = 0

    iter = 10000

    for index, (image, target) in enumerate(ds):

        plt.imshow(image.permute(1, 2, 0))
        plt.show()

        plt.imshow(image.permute(1, 2, 0))

        sequence = target['sequence']
        print(sequence[:, :5])

        for i, element in enumerate(sequence[1:]):
            if element[2 + CLASSES['<end-of-computation>']] == 1:
                break
            x = element[0] * 256
            y = element[1] * 256

            plt.scatter(x.cpu(), y.cpu(), label=i)

        plt.legend()
        plt.show()

        break

        if index % int(iter/100) == 0:
            print(index)

        if index > iter:
            print(f"No errors in {iter} iterations")
            break
        continue

    #     if index > 1000:
    #         break
    #     if index % 10 == 0:
    #         print(index)
    #
    #     sequence = target['sequence']
    #     classes = torch.argmax(sequence[:, 2:6], dim=1)
    #
    #     start += torch.where(classes == 0, 1, 0).sum().item()
    #     point += torch.where(classes == 1, 1, 0).sum().item()
    #     end_poly += torch.where(classes == 2, 1, 0).sum().item()
    #     end += torch.where(classes == 3, 1, 0).sum().item()
    #     tot += 26
    #     continue
    # print("START:", float(start)/tot)
    # print("POINT:", float(point)/tot)
    # print("END_POLY:", float(end_poly)/tot)
    # print("END:", float(end)/tot)

        # i = 0
        # while sequence[i][2 + CLASSES['<point>']] != 1:
        #     i += 1
        #
        # polygon = []
        # while sequence[i][2 + CLASSES['<end-of-computation>']] != 1:
        #     if sequence[i][2 + CLASSES['<point>']] != 1:
        #         for index, (x, y) in enumerate(polygon):
        #             plt.scatter(
        #                 [x*256], [y*256], label=str(index)
        #             )
        #         # plt.scatter([x*256 for x, y in polygon], [y*256 for x, y in polygon])
        #         # polygon = []
        #         break
        #     else:
        #         polygon.append((sequence[i][0], sequence[i][1]))
        #
        #     i += 1
        #
        # print('\n'.join(str(k) + ' --> ' + str(list(target[k].shape)) for k in target))
        #
        # plt.legend()
        # plt.show()
        #
        # break
