from toy_setting_open_cv.toy_setting import display_image, get_ts_image
from detr.model import Transformer

image, labels = get_ts_image(256, 256, rand_gate_number=True)

display_image(image, labels)

exit(-1)

model = Transformer(2)
model.eval()

CLASSES = ["N/A", "GATE"]
