import random
from math import sqrt


class Gate:

    def __init__(self, image_height, image_width, perc_of_image_width=0.80, color='none'):
        self.image_height = image_height
        self.image_width = image_width

        self.width = random.randint(0, int(self.image_width*perc_of_image_width))
        self.height = random.randint(
            int(self.width * 0.5),
            int(self.width * 1.5) if self.width * 1.5 < self.image_height else int(self.image_height)
        )

        self.stroke = random.randint(4, 10)

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
        return [
            self.bottom_left,
            self.top_left,
            self.top_right,
            self.bottom_right
        ]

    def rand_shift(self, perc=0.40):

        shift_h = random.randint(-int(self.image_width*perc), int(self.image_width*perc))
        shift_v = random.randint(-int(self.image_height*perc), int(self.image_height*perc))

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

    def rand_rotate(self, horizontal=False, incr_perc=0.30):

        if not horizontal:
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
