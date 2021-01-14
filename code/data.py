import torch.utils.data as t_data
from image_gen import get_ts_image
from torch import tensor


class TSDataset(t_data.Dataset):

    def __init__(self, img_height, img_width, num_gates=3, padding=5, rand_gate_number=True,
                 no_gate_chance=0.10, rotate_chance=0.5, shift_chance=0.5, transform=None):
        self.img_height = img_height
        self.img_width = img_width
        self.num_gates = num_gates
        self.padding = padding
        self.rand_gate_number = rand_gate_number
        self.no_gate_chance = no_gate_chance
        self.rotate_chance = rotate_chance
        self.shift_chance = shift_chance
        self.generated_images = 0
        self.transform = transform

    def __len__(self):
        return 1000000

    def __getitem__(self, index):
        self.generated_images += 1
        image, labels = get_ts_image(
            self.img_height,
            self.img_width,
            num_gates=self.num_gates,
            padding=self.padding,
            rand_gate_number=self.rand_gate_number,
            no_gate_chance=self.no_gate_chance,
            rotate_chance=self.rotate_chance,
            shift_chance=self.shift_chance
        )

        if self.transform:
            image = self.transform(image)
            labels = tensor(labels)

        return image, labels
