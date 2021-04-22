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
    "<end-of-polygon>": 2,
    "<end-of-computation>": 3
}

class PolyGate:

    MIN_CORNERS = 3
    MAX_CORNERS = 7

    def __init__(self, image_height, image_width, color='none', num_corners=-1, stroke=-1, clamp=True):

        self.image_height, self.image_width = image_height, image_width

        self.num_corners = num_corners

        self.c_x = random.randint(int(0.20*image_width), int(image_width - (0.20*image_width)))
        self.c_y = random.randint(int(0.20*image_height), int(image_height - (0.20*image_height)))

        radius_perc = random.uniform(0.05, 0.40)
        self.radius = image_height * radius_perc

        if num_corners == -1:
            num_corners = random.randint(self.MIN_CORNERS, self.MAX_CORNERS)

        self.corners = []

        alpha_zero = (2 * math.pi / num_corners)
        alpha = random.uniform(math.pi/2, math.pi)
        for _ in range(num_corners):
            x = self.c_x + self.radius * math.cos(alpha)
            y = self.c_y + self.radius * math.sin(alpha)
            slope = math.tan(alpha) if alpha != math.pi/2 else math.tan(alpha + 0.001)
            delta_x = random.uniform(0.0, math.sqrt(self.radius)/2)
            delta_y = slope * delta_x

            final_x = int(x+delta_x)
            final_y = int(y+delta_y)
            if clamp:
                final_x = 0 if final_x < 0 else final_x
                final_x = self.image_width if final_x > self.image_width else final_x

                final_y = 0 if final_y < 0 else final_y
                final_y = self.image_height if final_y > self.image_height else final_y

            self.corners.append((final_x, final_y))
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

        if self.num_corners == -1:
            while len(labels) != 2*self.MAX_CORNERS:
                labels.append(-1)

        return labels

    def get_area(self) -> float:
        return math.pi * (self.radius**2)

    def get_class(self) -> int:
        return len(self.corners)


def get_ts_image(height, width, num_gates=3, no_gate_chance=0.10, black_and_white=True, stroke=-1, num_corners=-1, clamp=True) -> (np.ndarray, list, list):

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
            gate = PolyGate(height, width, color='white', stroke=stroke, num_corners=num_corners, clamp=clamp)
        else:
            gate = PolyGate(height, width, color='none', stroke=stroke, num_corners=num_corners, clamp=clamp)

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
                    torch.clamp(gate[3], 0, h).item()
                ])
            target['boxes'] = torch.tensor(gates, dtype=torch.float32)
        return img, target


class GetSentence(object):
    def __init__(self, num_gates, num_corners):
        self.num_gates = num_gates
        self.num_corners = num_corners if num_corners != -1 else PolyGate.MAX_CORNERS

    def __call__(self, sample):
        img, target = sample

        max_lenght = (self.num_gates * (self.num_corners+1)) + 2

        centers = []
        for polygon in target['boxes']:
            index = len(polygon)
            for i, value in enumerate(polygon):
                if value == -1:
                    index = i
                    break
            mean_x = polygon[:index:2].mean().item()
            mean_y = polygon[1:index:2].mean().item()
            centers.append((mean_x, mean_y))
        centers = {i: c for i, c in enumerate(centers)}

        sequence = []
        start_token = torch.zeros(256)
        start_token[2 + CLASSES['<start>']] = 1
        sequence.append(start_token)

        while len(centers) > 0:
            min_center = [2, 2]
            for index in centers:
                x, y = centers[index]
                if y < min_center[1]:
                    min_center = (x, y)
                    min_index = index
                elif y == min_center[1] and x < min_center[0]:
                    min_center = (x, y)
                    min_index = index

            polygon = target['boxes'][min_index]
            for x, y in polygon.view(-1, 2):
                if x == -1:
                    break
                token = torch.zeros(256)
                token[2 + CLASSES['<point>']] = 1
                token[0] = x
                token[1] = y
                sequence.append(token)
            end_polygon = torch.zeros(256)
            end_polygon[2 + CLASSES['<end-of-polygon>']] = 1
            sequence.append(end_polygon)

            centers.pop(min_index)

        while len(sequence) < max_lenght:
            end_computation = torch.zeros(256)
            end_computation[2 + CLASSES['<end-of-computation>']] = 1
            sequence.append(end_computation)

        sequence = torch.stack(sequence)

        target['sequence'] = sequence
        return img, target


class MaskRCNN(object):
    def __call__(self, sample):
        img, target = sample

        h, w = target['size']
        h, w = h.item(), w.item()

        masks = []
        for label in target['boxes']:
            mask_img = Image.new('L', (w, h), 0)
            ImageDraw.Draw(mask_img).polygon([
                (label[0]*w, label[1]*h),
                (label[2]*w, label[3]*h),
                (label[4]*w, label[5]*h),
                (label[6]*w, label[7]*h),
            ], outline=1, fill=1)
            mask_img = torch.tensor(np.array(mask_img), dtype=torch.int8)
            masks.append(mask_img)
        target['masks'] = torch.stack(masks)

        gates = []
        for gate in target['boxes']:
            abs_gate = [
                gate[0].item() * w,
                gate[1].item() * h,
                gate[2].item() * w,
                gate[3].item() * h,
                gate[4].item() * w,
                gate[5].item() * h,
                gate[6].item() * w,
                gate[7].item() * h
            ]

            x0 = min([abs_gate[0], abs_gate[2], abs_gate[4], abs_gate[6]])
            x1 = max([abs_gate[0], abs_gate[2], abs_gate[4], abs_gate[6]])

            y0 = min([abs_gate[1], abs_gate[3], abs_gate[5], abs_gate[7]])
            y1 = max([abs_gate[1], abs_gate[3], abs_gate[5], abs_gate[7]])

            gates.append([x0, y0, x1, y1])

        target['boxes'] = torch.tensor(gates, dtype=torch.float32)

        return img, target


class TSDataset(torch.utils.data.Dataset):

    std_transform = T.Compose([
        ToTensor(),
        # Clamp()
    ])

    mask_transform = T.Compose([
        ToTensor(),
        MaskRCNN(),
        # Clamp()
    ])

    def __init__(self, img_height, img_width, num_gates=3, black_and_white=True,
                 no_gate_chance=0.0, stroke=-1, num_corners=4, mask=False, clamp_gates=False):
        self.img_height = img_height
        self.img_width = img_width
        self.num_gates = num_gates
        self.black_and_white = black_and_white
        self.no_gate_chance = no_gate_chance
        self.stroke = stroke
        self.num_corners = num_corners
        self.clamp_gates = clamp_gates
        if mask:
            self.transform = self.mask_transform
            self.label = 1
        else:
            self.transform = self.std_transform
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
            num_corners=self.num_corners,
            clamp=self.clamp_gates
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
            t = GetSentence(self.num_gates, self.num_corners)
            image, target = t((image, target))

        return image, target


if __name__ == '__main__':

    ds = TSDataset(256, 256, num_gates=3, black_and_white=True, no_gate_chance=0.0, stroke=-1, num_corners=-1, mask=False, clamp_gates=True)

    start = 0
    point = 0
    end_poly = 0
    end = 0

    tot = 0

    for index, (image, target) in enumerate(ds):
        plt.imshow(image.cpu().permute(1, 2, 0))
        plt.show()
        continue

        if index > 1000:
            break
        if index % 10 == 0:
            print(index)

        sequence = target['sequence']
        classes = torch.argmax(sequence[:, 2:6], dim=1)

        start += torch.where(classes == 0, 1, 0).sum().item()
        point += torch.where(classes == 1, 1, 0).sum().item()
        end_poly += torch.where(classes == 2, 1, 0).sum().item()
        end += torch.where(classes == 3, 1, 0).sum().item()
        tot += 26
        continue
    print("START:", float(start)/tot)
    print("POINT:", float(point)/tot)
    print("END_POLY:", float(end_poly)/tot)
    print("END:", float(end)/tot)

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
