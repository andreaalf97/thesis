import numpy as np
import cv2
import random
import os


class Gate:
    def __init__(self, image_height, image_width, perc_of_image_width=0.80, color='none'):
        self.image_height = image_height
        self.image_width = image_width

        self.width = random.randint(0, int(self.image_width*perc_of_image_width))
        self.height = random.randint(
            int(self.width * 0.5),
            int(self.width * 1.5) if self.width * 1.5 < self.image_height else int(self.image_height)
        )

        self.stroke = random.randint(1, 3)

        if 'white' in color.lower():
            self.color = (255, 255, 255)
        elif 'black' in color.lower():
            self.color = (0, 0, 0)
        else:
            self.color = (
                int(random.random() * 255),
                int(random.random() * 255),
                int(random.random() * 255)
            )

        self.bottom_left = (
            random.randint(0, self.image_width - self.width),
            random.randint(self.height, self.image_height)
        )

        self.top_left = (
            int(self.bottom_left[0]),
            int(self.bottom_left[1] - self.height)
        )

        self.top_right = (
            int(self.top_left[0] + self.width),
            int(self.top_left[1])
        )
        self.bottom_right = (
            int(self.top_right[0]),
            int(self.top_right[1] + self.height)
        )

    def get_labels(self) -> list:
        bl = (float(self.bottom_left[0])/self.image_width, float(self.bottom_left[1])/self.image_height)
        br = (float(self.bottom_right[0])/self.image_width, float(self.bottom_right[1])/self.image_height)
        tl = (float(self.top_left[0])/self.image_width, float(self.top_left[1])/self.image_height)
        tr = (float(self.top_right[0])/self.image_width, float(self.top_right[1])/self.image_height)

        return [
            bl,
            tl,
            tr,
            br
        ]

    def rand_shift(self, image_size_perc=0.40):

        shift_h = random.randint(-int(self.image_width * image_size_perc), int(self.image_width * image_size_perc))
        shift_v = random.randint(-int(self.image_height * image_size_perc), int(self.image_height * image_size_perc))

        self.bottom_left = (
            self.bottom_left[0] + shift_h,
            self.bottom_left[1] + shift_v
        )

        self.top_left = (
            self.top_left[0] + shift_h,
            self.top_left[1] + shift_v
        )

        self.top_right = (
            self.top_right[0] + shift_h,
            self.top_right[1] + shift_v
        )

        self.bottom_right = (
            self.bottom_right[0] + shift_h,
            self.bottom_right[1] + shift_v,
        )

    def rand_rotate(self, incr_perc=0.30):

        if random.random() > 0.5:
            increment = self.height * (random.random() * incr_perc)
            if random.random() > 0.5:
                increment = -increment

            self.bottom_left = (
                self.bottom_left[0],
                int(self.bottom_left[1] - increment/2)
            )

            self.top_left = (
                self.top_left[0],
                int(self.top_left[1] + increment/2)
            )

            self.top_right = (
                self.top_right[0],
                int(self.top_right[1] - increment/2)
            )

            self.bottom_right = (
                self.bottom_right[0],
                int(self.bottom_right[1] + increment/2),
            )
        else:
            increment = self.width * (random.random() * incr_perc)
            if random.random() > 0.5:
                increment = -increment

            self.bottom_left = (
                int(self.bottom_left[0] + increment/2),
                self.bottom_left[1]
            )

            self.top_left = (
                int(self.top_left[0] - increment/2),
                self.top_left[1]
            )

            self.top_right = (
                int(self.top_right[0] + increment/2),
                self.top_right[1]
            )

            self.bottom_right = (
                int(self.bottom_right[0] - increment/2),
                self.bottom_right[1],
            )


def get_ts_image(height, width, num_gates=3, padding=5, rand_gate_number=False, no_gate_chance=0.10, rotate_chance=0.5, shift_chance=0.5) -> (np.ndarray, list):
    assert padding >= 0 and isinstance(padding, int)

    if rand_gate_number:
        if random.random() < no_gate_chance:
            num_gates = 0
        else:
            num_gates = random.randint(1, num_gates)

    img = get_ts_background(height, width, bgr=True, real_background_prob=0.0, black_white=True)

    labels = []

    for _ in range(num_gates):
        if img[0][0][0] == 0:
            gate = Gate(height, width, perc_of_image_width=0.80, color='white')
        else:
            gate = Gate(height, width, perc_of_image_width=0.80, color='black')

        if random.random() < rotate_chance:
            gate.rand_rotate()
        if random.random() < shift_chance:
            gate.rand_shift(image_size_perc=0.20)
        labels.append(gate.get_labels())
        img = print_gate(img, gate, mark_top_corners=False)

    return img, labels


def print_gate(img: np.ndarray, gate: Gate, mark_top_corners=False) -> np.ndarray:

    img = cv2.line(
        img,
        gate.bottom_left,
        gate.top_left,
        gate.color,
        gate.stroke,
        lineType=cv2.LINE_AA
    )

    img = cv2.line(
        img,
        gate.top_left,
        gate.top_right,
        gate.color,
        gate.stroke,
        lineType=cv2.LINE_AA
    )

    img = cv2.line(
        img,
        gate.top_right,
        gate.bottom_right,
        gate.color,
        gate.stroke,
        lineType=cv2.LINE_AA
    )

    img = cv2.line(
        img,
        gate.bottom_right,
        gate.bottom_left,
        gate.color,
        gate.stroke,
        lineType=cv2.LINE_AA
    )

    if mark_top_corners:
        img = cv2.circle(
            img,
            gate.top_left,
            2,
            (255, 0, 0),
            -1
        )

        img = cv2.circle(
            img,
            gate.top_right,
            2,
            (0, 255, 0),
            -1
        )

    return img


def get_ts_background(height, width, real_background_prob=0.0, bgr=False, black_white=False) -> np.ndarray:

    coin = random.random()
    if coin <= real_background_prob:
        backgrounds = os.listdir("backgrounds")
        background = "backgrounds/" + random.choice(backgrounds)
        image = cv2.imread(background)
        image = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
        return image

    if black_white:
        image = np.zeros(
            (height, width, 3),
            dtype=np.uint8
        )

        # if random.random() > 0.5:
        #     image[:, :] = [0, 0, 0]
        # else:
        #     image[:, :] = [255, 255, 255]

        image[:, :] = [0, 0, 0]

        return image

    red = int(random.random() * 255)
    green = int(random.random() * 255)
    blue = int(random.random() * 255)

    # red = 0
    # green = 255
    # blue = 242

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
