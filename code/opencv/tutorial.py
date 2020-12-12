import numpy as np
import cv2

img = np.zeros((512, 512, 3), np.uint8)
img = cv2.line(
    img=img,
    pt1=(10, 10),
    pt2=(256, 400),
    color=(0, 0, 255),   # Color in this bullshit library are BGR (Blue - Green - Red)
    thickness=5,
    lineType=cv2.LINE_AA    # This uses an anti-aliased line
)


cv2.imshow("CUSTOM", img)
cv2.waitKey()
cv2.destroyAllWindows()