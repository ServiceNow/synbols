import pymunk
from pymunk.autogeometry import march_soft

from matplotlib import pyplot as plt

img = [
    "  xx   ",
    "  xx   ",
    "  xx   ",
    "  xx   ",
    "  xx   ",
    "  xxxxx",
    "  xxxxx",
]
segments = []

def segment_func(v0, v1):
    segments.append((tuple(v0), tuple(v1)))

def sample_func(point):
    x = int(point.x)
    y = int(point.y)
    return 1 if img[y][x] == "x" else 0

march_soft(pymunk.BB(0,0,6,6), 7, 7, .5, segment_func, sample_func)


for seg in segments:
    print(seg)