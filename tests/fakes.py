import numpy as np

from tms.core.types import Detection, PlateRead


def frame():
    return np.full((720, 1280, 3), 90, np.uint8)


class Scene:
    def __init__(self, riders=2, car=True):
        self.riders = riders
        self.car = car

    def __call__(self, image):
        found = [Detection(1, "motorcycle", (200, 300, 420, 560), 0.9)]
        for i in range(self.riders):
            left = 220 + i * 50
            found.append(Detection(10 + i, "person", (left, 120, left + 110, 520), 0.85))
        if self.car:
            found.append(Detection(2, "car", (700, 250, 1100, 560), 0.9))
        return found


def labeller(label, conf=0.95):
    return lambda crops: [(label, conf)] * len(crops)


def plate_reader(text, conf=0.9):
    return lambda vehicle: PlateRead(text, conf, vehicle[:40, :120])
