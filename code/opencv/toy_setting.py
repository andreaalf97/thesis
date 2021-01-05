import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import os
from gate import Gate




def get_ts_image(height, width, max_gates=3, padding=5) -> (np.ndarray, list):
    assert padding >= 0 and isinstance(padding, int)

    img = get_ts_background(height, width, bgr=True, real_background_prob=0.0, black_white=True)

    if img[0][0][0] == 0:
        gate = Gate(height, width, perc_of_image_width=0.80, color='white')
    else:
        gate = Gate(height, width, perc_of_image_width=0.80, color='black')

    # img = print_gate(img, gate, mark_top_corners=True)
    gate.rand_rotate(horizontal=False)

    # img = print_gate(img, gate)

    # gate.rand_shift(perc=0.20)

    img = print_gate(img, gate, mark_top_corners=True)

    return img, gate.get_labels()


def print_gate(img:np.ndarray, gate: Gate, mark_top_corners=False) -> np.ndarray:

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

        if random.random() > 0.5:
            image[:, :] = [0, 0, 0]
        else:
            image[:, :] = [255, 255, 255]

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


if __name__ == '__main__':
    image, labels = get_ts_image(540, 960)

    cv2.imwrite("test.png", image)

    cv2.imshow("THERE ARE {} GATES".format(len(labels)), image)
    cv2.waitKey()
    cv2.destroyAllWindows()

# img = cv2.line(
#     img=img,
#     pt1=(10, 10),
#     pt2=(256, 400),
#     color=(0, 0, 255),   # Color in this bullshit library are BGR (Blue - Green - Red)
#     thickness=5,
#     lineType=cv2.LINE_AA    # This uses an anti-aliased line
# )