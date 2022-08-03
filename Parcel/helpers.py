import operator
from functools import reduce
from random import sample
import cv2
import numpy as np
import tensorflow as tf


def extend_list(lol):
    if len(lol) >= 20:
        lol = sample(lol, 10)
    else:
        lol.extend((10 - len(lol)) * [[0, 0]])
    lol = np.array([(np.array(i).flatten()) for i in lol]).flatten()
    return lol


def interpolate(lol, n=1000, t="linear"):
    if len(lol) == n:
        return lol
    elif len(lol) < n:
        final_x = []
        final_y = []
        x = [point[0] for point in lol]
        y = [point[1] for point in lol]
        x.append(x[0])
        y.append(y[0])
        n_to_inter_most = int(n / len(lol))
        n_to_inter_last = n % len(lol)
        for x_r in range(len(x) - 1):
            x1, y1, x2, y2 = x[x_r], y[x_r], x[x_r + 1], y[x_r + 1]
            if x_r == len(x) - 2:
                steps = np.linspace(0, 1, n_to_inter_most + n_to_inter_last + 1)
            else:
                steps = np.linspace(0, 1, n_to_inter_most + 1)
            if t == "same":
                current_x_to_interpolate = [x1] * len(steps)
                current_y_to_interpolate = [y1] * len(steps)
            else:
                current_x_to_interpolate = [(x1 + (x2 - x1) * (step)) for step in steps]
                current_y_to_interpolate = [(y1 + (y2 - y1) * (step)) for step in steps]
            final_x.extend(current_x_to_interpolate[:-1])
            final_y.extend(current_y_to_interpolate[:-1])
        lol = np.array([np.array([final_x[pt], final_y[pt]]) for pt in range(len(final_x))])
    return lol


def flatten(lol):
    lol = np.array([(np.array(i).flatten()) for i in lol]).flatten()
    return lol


def bbox(lol):
    x = [pt[0] for pt in lol]
    y = [pt[1] for pt in lol]
    return np.array([min(x), min(y), max(x), max(y)])


def center(lol):
    center = list(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), lol), [len(lol)] * 2))
    return np.array(center)


def find_area(coords):
    coords = np.rint(coords).astype(np.int32)
    img = cv2.fillPoly(np.zeros((256, 256)), [np.int32(coords)], (255, 0, 0))
    return len((np.where(img == 255))[0])


def load_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_png(byte_img)
    return img


def distance(l1, l2=[0, 0]):
    d = ((l1[0] - l2[0]) ** 2 + (l1[1] - l2[1]) ** 2) ** 0.5
    return d


def sort_coords(coords):
    dst = list(map(distance, coords))
    origin = dst.index(min(dst))
    final_coords = coords[origin:] + coords[:origin]
    return final_coords


# def four_corners(lol):
#     top_left_dst = list(map(lambda x: distance(x, [0, 0]), lol))
#     left_center_half_min_1_dst = list(map(lambda x: distance(x, [0, 64]), lol))
#     left_center_dst = list(map(lambda x: distance(x, [0, 128]), lol))
#     left_center_half_max_1_dst = list(map(lambda x: distance(x, [0, 192]), lol))
#     bottom_left_dst = list(map(lambda x: distance(x, [0, 256]), lol))
#     bottom_center_half_min_1_dst = list(map(lambda x: distance(x, [64, 256]), lol))
#     bottom_center_dst = list(map(lambda x: distance(x, [128, 256]), lol))
#     bottom_center_half_max_1_dst = list(map(lambda x: distance(x, [192, 256]), lol))
#     bottom_right_dst = list(map(lambda x: distance(x, [256, 256]), lol))
#     right_center_half_max_1_dst = list(map(lambda x: distance(x, [256, 192]), lol))
#     right_center_dst = list(map(lambda x: distance(x, [256, 128]), lol))
#     right_center_half_min_1_dst = list(map(lambda x: distance(x, [256, 64]), lol))
#     top_right_dst = list(map(lambda x: distance(x, [256, 0]), lol))
#     top_center_half_max_1_dst = list(map(lambda x: distance(x, [192, 0]), lol))
#     top_center_dst = list(map(lambda x: distance(x, [128, 0]), lol))
#     top_center_half_min_1_dst = list(map(lambda x: distance(x, [64, 0]), lol))
#     top_left = lol[top_left_dst.index(min(top_left_dst))]
#     left_center_half_min_1 = lol[left_center_half_min_1_dst.index(min(left_center_half_min_1_dst))]
#     left_center = lol[left_center_dst.index(min(left_center_dst))]
#     left_center_half_max_1 = lol[left_center_half_max_1_dst.index(min(left_center_half_max_1_dst))]
#     bottom_left = lol[bottom_left_dst.index(min(bottom_left_dst))]
#     bottom_center_half_min_1 = lol[bottom_center_half_min_1_dst.index(min(bottom_center_half_min_1_dst))]
#     bottom_center = lol[bottom_center_dst.index(min(bottom_center_dst))]
#     bottom_center_half_max_1 = lol[bottom_center_half_max_1_dst.index(min(bottom_center_half_max_1_dst))]
#     bottom_right = lol[bottom_right_dst.index(min(bottom_right_dst))]
#     right_center_half_max = lol[right_center_half_max_1_dst.index(min(right_center_half_max_1_dst))]
#     right_center = lol[right_center_dst.index(min(right_center_dst))]
#     right_center_half_min = lol[right_center_half_min_1_dst.index(min(right_center_half_min_1_dst))]
#     top_right = lol[top_right_dst.index(min(top_right_dst))]
#     top_center_half_max = lol[top_center_half_max_1_dst.index(min(top_center_half_max_1_dst))]
#     top_center = lol[top_center_dst.index(min(top_center_dst))]
#     top_center_half_min = lol[top_center_half_min_1_dst.index(min(top_center_half_min_1_dst))]
#     return np.array(
#         [
#             top_left,
#             left_center_half_min_1,
#             left_center,
#             left_center_half_max_1,
#             bottom_left,
#             bottom_center_half_min_1,
#             bottom_center,
#             bottom_center_half_max_1,
#             bottom_right,
#             right_center_half_max,
#             right_center,
#             right_center_half_min,
#             top_right,
#             top_center_half_max,
#             top_center,
#             top_center_half_min,
#             top_left,
#         ]
#     )


# def four_corners(lol):
#     top_left_dst = list(map(lambda x: distance(x, [0,0]), lol))
#     left_center_dst = list(map(lambda x: distance(x, [0, 128]), lol))
#     bottom_left_dst = list(map(lambda x: distance(x, [0, 256]), lol))
#     bottom_center_dst = list(map(lambda x: distance(x, [128, 256]), lol))
#     bottom_right_dst = list(map(lambda x: distance(x, [256, 256]), lol))
#     right_center_dst = list(map(lambda x: distance(x, [256, 128]), lol))
#     top_right_dst = list(map(lambda x: distance(x, [256, 0]), lol))
#     top_center_dst = list(map(lambda x: distance(x, [128, 0]), lol))
#     top_left = lol[top_left_dst.index(min(top_left_dst))]
#     left_center = lol[left_center_dst.index(min(left_center_dst))]
#     bottom_left = lol[bottom_left_dst.index(min(bottom_left_dst))]
#     bottom_center = lol[bottom_center_dst.index(min(bottom_center_dst))]
#     bottom_right = lol[bottom_right_dst.index(min(bottom_right_dst))]
#     right_center = lol[right_center_dst.index(min(right_center_dst))]
#     top_right = lol[top_right_dst.index(min(top_right_dst))]
#     top_center = lol[top_center_dst.index(min(top_center_dst))]
#     return np.array([top_left, left_center, bottom_left, bottom_center, bottom_right, right_center, top_right, top_center, top_left])


# def four_corners(lol):
#     top_left_dst = list(map(lambda x: distance(x, [0,0]), lol))
#     bottom_left_dst = list(map(lambda x: distance(x, [0, 256]), lol))
#     bottom_right_dst = list(map(lambda x: distance(x, [256, 256]), lol))
#     top_right_dst = list(map(lambda x: distance(x, [256, 0]), lol))
#     top_left = lol[top_left_dst.index(min(top_left_dst))]
#     bottom_left = lol[bottom_left_dst.index(min(bottom_left_dst))]
#     bottom_right = lol[bottom_right_dst.index(min(bottom_right_dst))]
#     top_right = lol[top_right_dst.index(min(top_right_dst))]
#     return np.array([top_left, bottom_left, bottom_right, top_right, top_left])

# def four_corners(lol):
#     top_left_dst = list(map(lambda x: distance(x, [0, 0]), lol))
#     left_center_half_min_2_dst = list(map(lambda x: distance(x, [0, 32]), lol))
#     left_center_half_min_1_dst = list(map(lambda x: distance(x, [0, 64]), lol))
#     left_center_half_min_3_dst = list(map(lambda x: distance(x, [0, 96]), lol))
#     left_center_dst = list(map(lambda x: distance(x, [0, 128]), lol))
#     left_center_half_max_2_dst = list(map(lambda x: distance(x, [0, 160]), lol))
#     left_center_half_max_1_dst = list(map(lambda x: distance(x, [0, 192]), lol))
#     left_center_half_max_3_dst = list(map(lambda x: distance(x, [0, 224]), lol))
#     bottom_left_dst = list(map(lambda x: distance(x, [0, 256]), lol))
#     bottom_center_half_min_2_dst = list(map(lambda x: distance(x, [32, 256]), lol))
#     bottom_center_half_min_1_dst = list(map(lambda x: distance(x, [64, 256]), lol))
#     bottom_center_half_min_3_dst = list(map(lambda x: distance(x, [96, 256]), lol))
#     bottom_center_dst = list(map(lambda x: distance(x, [128, 256]), lol))
#     bottom_center_half_max_2_dst = list(map(lambda x: distance(x, [160, 256]), lol))
#     bottom_center_half_max_1_dst = list(map(lambda x: distance(x, [192, 256]), lol))
#     bottom_center_half_max_3_dst = list(map(lambda x: distance(x, [224, 256]), lol))
#     bottom_right_dst = list(map(lambda x: distance(x, [256, 256]), lol))
#     right_center_half_max_2_dst = list(map(lambda x: distance(x, [256, 224]), lol))
#     right_center_half_max_1_dst = list(map(lambda x: distance(x, [256, 192]), lol))
#     right_center_half_max_3_dst = list(map(lambda x: distance(x, [256, 160]), lol))
#     right_center_dst = list(map(lambda x: distance(x, [256, 128]), lol))
#     right_center_half_min_2_dst = list(map(lambda x: distance(x, [256, 96]), lol))
#     right_center_half_min_1_dst = list(map(lambda x: distance(x, [256, 64]), lol))
#     right_center_half_min_3_dst = list(map(lambda x: distance(x, [256, 32]), lol))
#     top_right_dst = list(map(lambda x: distance(x, [256, 0]), lol))
#     top_center_half_max_2_dst = list(map(lambda x: distance(x, [224, 0]), lol))
#     top_center_half_max_1_dst = list(map(lambda x: distance(x, [192, 0]), lol))
#     top_center_half_max_3_dst = list(map(lambda x: distance(x, [160, 0]), lol))
#     top_center_dst = list(map(lambda x: distance(x, [128, 0]), lol))
#     top_center_half_min_2_dst = list(map(lambda x: distance(x, [96, 0]), lol))
#     top_center_half_min_1_dst = list(map(lambda x: distance(x, [64, 0]), lol))
#     top_center_half_min_3_dst = list(map(lambda x: distance(x, [32, 0]), lol))
#     top_left = lol[top_left_dst.index(min(top_left_dst))]
#     left_center_half_min_2 = lol[left_center_half_min_2_dst.index(min(left_center_half_min_2_dst))]
#     left_center_half_min_1 = lol[left_center_half_min_1_dst.index(min(left_center_half_min_1_dst))]
#     left_center_half_min_3 = lol[left_center_half_min_3_dst.index(min(left_center_half_min_3_dst))]
#     left_center = lol[left_center_dst.index(min(left_center_dst))]
#     left_center_half_max_2 = lol[left_center_half_max_2_dst.index(min(left_center_half_max_2_dst))]
#     left_center_half_max_1 = lol[left_center_half_max_1_dst.index(min(left_center_half_max_1_dst))]
#     left_center_half_max_3 = lol[left_center_half_max_3_dst.index(min(left_center_half_max_3_dst))]
#     bottom_left = lol[bottom_left_dst.index(min(bottom_left_dst))]
#     bottom_center_half_min_2 = lol[bottom_center_half_min_2_dst.index(min(bottom_center_half_min_2_dst))]
#     bottom_center_half_min_1 = lol[bottom_center_half_min_1_dst.index(min(bottom_center_half_min_1_dst))]
#     bottom_center_half_min_3 = lol[bottom_center_half_min_3_dst.index(min(bottom_center_half_min_3_dst))]
#     bottom_center = lol[bottom_center_dst.index(min(bottom_center_dst))]
#     bottom_center_half_max_2 = lol[bottom_center_half_max_2_dst.index(min(bottom_center_half_max_2_dst))]
#     bottom_center_half_max_1 = lol[bottom_center_half_max_1_dst.index(min(bottom_center_half_max_1_dst))]
#     bottom_center_half_max_3 = lol[bottom_center_half_max_3_dst.index(min(bottom_center_half_max_3_dst))]
#     bottom_right = lol[bottom_right_dst.index(min(bottom_right_dst))]
#     right_center_half_max_2 = lol[right_center_half_max_2_dst.index(min(right_center_half_max_2_dst))]
#     right_center_half_max_1 = lol[right_center_half_max_1_dst.index(min(right_center_half_max_1_dst))]
#     right_center_half_max_3 = lol[right_center_half_max_3_dst.index(min(right_center_half_max_3_dst))]
#     right_center = lol[right_center_dst.index(min(right_center_dst))]
#     right_center_half_min_2 = lol[right_center_half_min_2_dst.index(min(right_center_half_min_2_dst))]
#     right_center_half_min_1 = lol[right_center_half_min_1_dst.index(min(right_center_half_min_1_dst))]
#     right_center_half_min_3 = lol[right_center_half_min_3_dst.index(min(right_center_half_min_3_dst))]
#     top_right = lol[top_right_dst.index(min(top_right_dst))]
#     top_center_half_max_2 = lol[top_center_half_max_2_dst.index(min(top_center_half_max_2_dst))]
#     top_center_half_max_1 = lol[top_center_half_max_1_dst.index(min(top_center_half_max_1_dst))]
#     top_center_half_max_3 = lol[top_center_half_max_3_dst.index(min(top_center_half_max_3_dst))]
#     top_center = lol[top_center_dst.index(min(top_center_dst))]
#     top_center_half_min_2 = lol[top_center_half_min_2_dst.index(min(top_center_half_min_2_dst))]
#     top_center_half_min_1 = lol[top_center_half_min_1_dst.index(min(top_center_half_min_1_dst))]
#     top_center_half_min_3 = lol[top_center_half_min_3_dst.index(min(top_center_half_min_3_dst))]
#     return np.array(
#         [
#             top_left,
#             left_center_half_min_2,
#             left_center_half_min_1,
#             left_center_half_min_3,
#             left_center,
#             left_center_half_max_2,
#             left_center_half_max_1,
#             left_center_half_max_3,
#             bottom_left,
#             bottom_center_half_min_2,
#             bottom_center_half_min_1,
#             bottom_center_half_min_3,
#             bottom_center,
#             bottom_center_half_max_2,
#             bottom_center_half_max_1,
#             bottom_center_half_max_3,
#             bottom_right,
#             right_center_half_max_2,
#             right_center_half_max_1,
#             right_center_half_max_3,
#             right_center,
#             right_center_half_min_2,
#             right_center_half_min_1,
#             right_center_half_min_3,
#             top_right,
#             top_center_half_max_2,
#             top_center_half_max_1,
#             top_center_half_max_3,
#             top_center,
#             top_center_half_min_2,
#             top_center_half_min_1,
#             top_center_half_min_3,
#             top_left,
#         ]
#     )


def four_corners(lol):
    top_left_dst = list(map(lambda x: distance(x, [0, 0]), lol))
    left_center_half_min_2_dst = list(map(lambda x: distance(x, [0, 32]), lol))
    left_center_half_min_1_dst = list(map(lambda x: distance(x, [0, 64]), lol))
    left_center_half_min_3_dst = list(map(lambda x: distance(x, [0, 96]), lol))
    left_center_dst = list(map(lambda x: distance(x, [0, 128]), lol))
    left_center_half_max_2_dst = list(map(lambda x: distance(x, [0, 160]), lol))
    left_center_half_max_1_dst = list(map(lambda x: distance(x, [0, 192]), lol))
    left_center_half_max_3_dst = list(map(lambda x: distance(x, [0, 224]), lol))
    bottom_left_dst = list(map(lambda x: distance(x, [0, 256]), lol))
    bottom_center_half_min_2_dst = list(map(lambda x: distance(x, [32, 256]), lol))
    bottom_center_half_min_1_dst = list(map(lambda x: distance(x, [64, 256]), lol))
    bottom_center_half_min_3_dst = list(map(lambda x: distance(x, [96, 256]), lol))
    bottom_center_dst = list(map(lambda x: distance(x, [128, 256]), lol))
    bottom_center_half_max_2_dst = list(map(lambda x: distance(x, [160, 256]), lol))
    bottom_center_half_max_1_dst = list(map(lambda x: distance(x, [192, 256]), lol))
    bottom_center_half_max_3_dst = list(map(lambda x: distance(x, [224, 256]), lol))
    bottom_right_dst = list(map(lambda x: distance(x, [256, 256]), lol))
    right_center_half_max_2_dst = list(map(lambda x: distance(x, [256, 224]), lol))
    right_center_half_max_1_dst = list(map(lambda x: distance(x, [256, 192]), lol))
    right_center_half_max_3_dst = list(map(lambda x: distance(x, [256, 160]), lol))
    right_center_dst = list(map(lambda x: distance(x, [256, 128]), lol))
    right_center_half_min_2_dst = list(map(lambda x: distance(x, [256, 96]), lol))
    right_center_half_min_1_dst = list(map(lambda x: distance(x, [256, 64]), lol))
    right_center_half_min_3_dst = list(map(lambda x: distance(x, [256, 32]), lol))
    top_right_dst = list(map(lambda x: distance(x, [256, 0]), lol))
    top_center_half_max_2_dst = list(map(lambda x: distance(x, [224, 0]), lol))
    top_center_half_max_1_dst = list(map(lambda x: distance(x, [192, 0]), lol))
    top_center_half_max_3_dst = list(map(lambda x: distance(x, [160, 0]), lol))
    top_center_dst = list(map(lambda x: distance(x, [128, 0]), lol))
    top_center_half_min_2_dst = list(map(lambda x: distance(x, [96, 0]), lol))
    top_center_half_min_1_dst = list(map(lambda x: distance(x, [64, 0]), lol))
    top_center_half_min_3_dst = list(map(lambda x: distance(x, [32, 0]), lol))
    top_left = lol[top_left_dst.index(min(top_left_dst))]
    left_center_half_min_2 = lol[left_center_half_min_2_dst.index(min(left_center_half_min_2_dst))]
    left_center_half_min_1 = lol[left_center_half_min_1_dst.index(min(left_center_half_min_1_dst))]
    left_center_half_min_3 = lol[left_center_half_min_3_dst.index(min(left_center_half_min_3_dst))]
    left_center = lol[left_center_dst.index(min(left_center_dst))]
    left_center_half_max_2 = lol[left_center_half_max_2_dst.index(min(left_center_half_max_2_dst))]
    left_center_half_max_1 = lol[left_center_half_max_1_dst.index(min(left_center_half_max_1_dst))]
    left_center_half_max_3 = lol[left_center_half_max_3_dst.index(min(left_center_half_max_3_dst))]
    bottom_left = lol[bottom_left_dst.index(min(bottom_left_dst))]
    bottom_center_half_min_2 = lol[bottom_center_half_min_2_dst.index(min(bottom_center_half_min_2_dst))]
    bottom_center_half_min_1 = lol[bottom_center_half_min_1_dst.index(min(bottom_center_half_min_1_dst))]
    bottom_center_half_min_3 = lol[bottom_center_half_min_3_dst.index(min(bottom_center_half_min_3_dst))]
    bottom_center = lol[bottom_center_dst.index(min(bottom_center_dst))]
    bottom_center_half_max_2 = lol[bottom_center_half_max_2_dst.index(min(bottom_center_half_max_2_dst))]
    bottom_center_half_max_1 = lol[bottom_center_half_max_1_dst.index(min(bottom_center_half_max_1_dst))]
    bottom_center_half_max_3 = lol[bottom_center_half_max_3_dst.index(min(bottom_center_half_max_3_dst))]
    bottom_right = lol[bottom_right_dst.index(min(bottom_right_dst))]
    right_center_half_max_2 = lol[right_center_half_max_2_dst.index(min(right_center_half_max_2_dst))]
    right_center_half_max_1 = lol[right_center_half_max_1_dst.index(min(right_center_half_max_1_dst))]
    right_center_half_max_3 = lol[right_center_half_max_3_dst.index(min(right_center_half_max_3_dst))]
    right_center = lol[right_center_dst.index(min(right_center_dst))]
    right_center_half_min_2 = lol[right_center_half_min_2_dst.index(min(right_center_half_min_2_dst))]
    right_center_half_min_1 = lol[right_center_half_min_1_dst.index(min(right_center_half_min_1_dst))]
    right_center_half_min_3 = lol[right_center_half_min_3_dst.index(min(right_center_half_min_3_dst))]
    top_right = lol[top_right_dst.index(min(top_right_dst))]
    top_center_half_max_2 = lol[top_center_half_max_2_dst.index(min(top_center_half_max_2_dst))]
    top_center_half_max_1 = lol[top_center_half_max_1_dst.index(min(top_center_half_max_1_dst))]
    top_center_half_max_3 = lol[top_center_half_max_3_dst.index(min(top_center_half_max_3_dst))]
    top_center = lol[top_center_dst.index(min(top_center_dst))]
    top_center_half_min_2 = lol[top_center_half_min_2_dst.index(min(top_center_half_min_2_dst))]
    top_center_half_min_1 = lol[top_center_half_min_1_dst.index(min(top_center_half_min_1_dst))]
    top_center_half_min_3 = lol[top_center_half_min_3_dst.index(min(top_center_half_min_3_dst))]
    return np.array(
        [
            top_left,
            left_center_half_min_2,
            left_center_half_min_1,
            left_center_half_min_3,
            left_center,
            left_center_half_max_2,
            left_center_half_max_1,
            left_center_half_max_3,
            bottom_left,
            bottom_center_half_min_2,
            bottom_center_half_min_1,
            bottom_center_half_min_3,
            bottom_center,
            bottom_center_half_max_2,
            bottom_center_half_max_1,
            bottom_center_half_max_3,
            bottom_right,
            right_center_half_max_2,
            right_center_half_max_1,
            right_center_half_max_3,
            right_center,
            right_center_half_min_2,
            right_center_half_min_1,
            right_center_half_min_3,
            top_right,
            top_center_half_max_2,
            top_center_half_max_1,
            top_center_half_max_3,
            top_center,
            top_center_half_min_2,
            top_center_half_min_1,
            top_center_half_min_3,
            top_left,
        ]
    )
