import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import os


def get_ts_image(height, width, max_gates=3, padding=5) -> (np.ndarray, list):
    assert padding >= 0 and isinstance(padding, int)

    img = get_ts_background(height, width, bgr=True, real_background_prob=1.0)

    x_range = (-padding, width+padding)
    y_range = (-padding, height+padding)
    if random.random() < 0.05:
        num_gates = 0
    else:
        num_gates = random.randint(1, max_gates)

    gate_labels = []

    for _ in range(num_gates):
        color = (
            random.random() * 255,
            random.random() * 255,
            random.random() * 255
        )

        stroke = random.randint(4, 50)

        top_left = (
            int(random.random() * (x_range[1]-x_range[0]) + x_range[0]),
            int(random.random() * (y_range[1] - y_range[0]) + y_range[0])
        )

        if top_left[0] > width/2:   # If the top left point is in the right hand side of the image, we go left
            top_right = (
                top_left[0] - abs(int(random.random() * (width) - (width/2))),
                top_left[1] + int(random.random() * (height) - (height/2)),
            )
        else:
            top_right = (
                top_left[0] + abs(int(random.random() * (width) - (width / 2))),
                top_left[1] + int(random.random() * (height) - (height / 2)),
            )

        if top_left[1] > height/2:  # If the top left is in the bottom of the image, we go up
            bottom_left = (
                top_left[0],
                top_left[1] - random.randint(int(-height/2), 0)
            )
        else:
            bottom_left = (
                top_left[0],
                top_left[1] + random.randint(0, int(height/2))
            )

        bottom_right = (
            top_right[0],
            bottom_left[1] + (top_right[1]-top_left[1])
        )

        img = cv2.line(
            img,
            bottom_left,
            top_left,
            color,
            stroke,
            lineType=cv2.LINE_AA
        )

        img = cv2.line(
            img,
            top_left,
            top_right,
            color,
            stroke,
            lineType=cv2.LINE_AA
        )

        img = cv2.line(
            img,
            top_right,
            bottom_right,
            color,
            stroke,
            lineType=cv2.LINE_AA
        )

        img = cv2.line(
            img,
            bottom_right,
            bottom_left,
            color,
            stroke,
            lineType=cv2.LINE_AA
        )

        img = cv2.circle(
            img,
            top_left,
            2,
            (255, 0, 0),
            -1
        )

        img = cv2.circle(
            img,
            top_right,
            2,
            (0, 255, 0),
            -1
        )

        gate_labels.append(
            (top_left, top_right, bottom_right, bottom_left)
        )

    return img, gate_labels


def get_ts_background(height, width, real_background_prob=0.0, bgr=False) -> np.ndarray:

    coin = random.random()
    if coin <= real_background_prob:
        backgrounds = os.listdir("backgrounds")
        background = "backgrounds/" + random.choice(backgrounds)
        image = cv2.imread(background)
        image = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
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
    image, labels = get_ts_image(540, 860)

    # cv2.imwrite("samples/test.png", i)

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