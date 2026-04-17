# mask_to_polygon.py

import cv2

def mask_to_polygon(mask):
    contours, _ = cv2.findContours(mask.astype("uint8"),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    polygon = []
    for c in contours:
        for point in c:
            x, y = point[0]
            polygon.append((x, y))

    return polygon