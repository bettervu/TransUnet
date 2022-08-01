import cv2
import math
import gcsfs
import numpy as np
import pandas as pd
from PIL import Image
from shapely import wkt
from bp.utils import add_geom_to_img
from shapely.geometry import Polygon
from bp.database import Gtu, db_session
from shapely.geometry import MultiPolygon
from bp.bounding_box import BoundingBox, polygon_to_pixel
from bp.enums import ImageryProviderTypes, InputImageTypes
from bp.boundary.utils import get_north_image_shape, get_oblique_poly, extend_property_poly
from bp.utils.helpers import polygon_coords_to_list, get_interiors_coords_list, geom_to_poly, coords_to_valid_poly
from dTurk.utils.process_layers import resize_or_pad
from dTurk.utils.helpers import gsd_normalize

FS = gcsfs.GCSFileSystem()

with db_session() as sess:
    records = sess.query(Gtu).offset(150000).limit(30000).all()


def possible_image_location(property_id, gtu_id):
    gtu_id_dir = f"bvds/GTU/{property_id}/{gtu_id}/"
    property_id_dir = f"bvds/GTU/{property_id}/"
    return gtu_id_dir, property_id_dir


def open_image(gtu_id_dir: str, property_id_dir: str, reference_image: str):
    gtu_id_dir += reference_image
    property_id_dir += reference_image
    try:
        with FS.open(gtu_id_dir, "rb") as f:
            img = np.array(Image.open(f))
            return img
    except FileNotFoundError:
        try:
            with FS.open(property_id_dir, "rb") as f:
                img = np.array(Image.open(f))
                return img
        except FileNotFoundError:
            print("File not in location")


def slope(x1, y1, x2, y2):
    m = (y2 - y1) / (x2 - x1)
    return m


def distance(x1, y1, x2, y2):
    d = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    return d


def angle(a, b, c):
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return ang + 360 if ang < 0 else ang


def shift_coords_crop(coords, og_shape, new_shape):
    for coord in coords:
        coord[1] = coord[1] * (new_shape[0] / og_shape[0])
        coord[0] = coord[0] * (new_shape[1] / og_shape[1])
    return coords


def shift_coords_roi(coords, og_shape, x_min=None, y_min=None):
    for coord in coords:
        coord[0] = coord[0] - x_min
        coord[1] = coord[1] - y_min
    return coords


def shift_coords_pad(coords, og_shape, crop_size=256):
    p_x = (crop_size - og_shape[1]) // 2
    p_y = (crop_size - og_shape[0]) // 2
    for coord in coords:
        coord[0] = coord[0] + p_x
        coord[1] = coord[1] + p_y
    return coords


def reshift_coords(img1, property_mask1, coords, init_gsd):
    img2, (
        image_height_1,
        image_width_1,
    ) = gsd_normalize(img1, original_gsd=init_gsd, final_gsd=10)
    property_mask1, _ = gsd_normalize(property_mask1, original_gsd=init_gsd, final_gsd=10)
    coords = shift_coords_crop(coords, img1.shape, (image_height_1, image_width_1))

    (
        img4,
        (start_padding_height, end_padding_height, start_padding_width, end_padding_width),
        (image_height_2, image_width_2),
    ) = resize_or_pad(img2, 256)
    coords = shift_coords_crop(coords, img2.shape, (image_height_2, image_width_2))
    coords = shift_coords_pad(coords, (image_height_2, image_width_2))
    return img4, coords


def cleanup1(coords):
    point = 0
    m = slope(coords[0][0], coords[0][1], coords[-1][0], coords[-1][1])
    d = distance(coords[0][0], coords[0][1], coords[-1][0], coords[-1][1])
    if d == 0:
        coords.pop()
    while point < len(coords) - 1:
        m = slope(coords[point][0], coords[point][1], coords[point + 1][0], coords[point + 1][1])
        d = distance(coords[point][0], coords[point][1], coords[point + 1][0], coords[point + 1][1])
        if d < 3:
            coords.pop(point + 1)
        else:
            point += 1
    return coords


def cleanup2(coords):
    point = 0
    cache_slope = 0
    while point < len(coords) - 1:
        m = slope(coords[point][0], coords[point][1], coords[point + 1][0], coords[point + 1][1])
        d = distance(coords[point][0], coords[point][1], coords[point + 1][0], coords[point + 1][1])
        if abs(m) == abs(cache_slope):
            coords.pop(point)
        else:
            cache_slope = m
            point += 1
    return coords


def cleanup3(coords):
    point = 1
    while point < len(coords) - 1:
        p0x = coords[point - 1][0]
        p0y = coords[point - 1][1]
        p1x = coords[point][0]
        p1y = coords[point][1]
        p2x = coords[point + 1][0]
        p2y = coords[point + 1][1]

        ang = angle([p0x, p0y], [p1x, p1y], [p2x, p2y])

        if abs(ang - 180) < 12.5 or abs(ang - 360) < 12.5 or ang < 12.5:
            coords.pop(point)
        elif type(ang):
            point += 1
        else:
            point += 1
    return coords


def add_geom_to_img(img, geom, color=(1,), thickness=-1, modify_img=True):
    def add_poly_to_img(img, poly, color, thickness):
        coords = np.rint(polygon_coords_to_list(poly)).astype(np.int32)
        holes = [np.rint(xy).astype(np.int32) for xy in get_interiors_coords_list(poly)]
        return cv2.drawContours(img, [coords] + holes, -1, color, thickness), [coords] + holes

    if not modify_img:
        img = img.copy()

    if isinstance(geom, Polygon):
        if not geom.is_empty:
            img, coord = add_poly_to_img(img, geom, color, thickness)
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            if not poly.is_empty:
                img, coord = add_poly_to_img(img, poly, color, thickness)
    elif isinstance(geom, list):
        for poly in geom:
            if poly and isinstance(poly, Polygon) and not poly.is_empty:
                img, coord = add_poly_to_img(img, poly, color, thickness)

    return img, coord


def get_property_info(
    bbox: BoundingBox,
    image_shape,
    property_data,
    input_image_type_id=InputImageTypes.overhead(),
    input_image_provider_type_id=None,
):
    if (
        ImageryProviderTypes.is_nearmap(input_image_provider_type_id)
        or ImageryProviderTypes.is_vexcel(input_image_provider_type_id)
    ) and InputImageTypes.is_oblique(input_image_type_id):
        north_image_shape = get_north_image_shape(input_image_type_id, image_shape)
        north_lnglat_property_geom = wkt.loads(property_data)
        north_property_geom = polygon_to_pixel(north_lnglat_property_geom, bbox, north_image_shape)
        north_property_poly = geom_to_poly(north_property_geom)
        property_poly = get_oblique_poly(north_property_poly, input_image_type_id, north_image_shape)
        property_poly = extend_property_poly(property_poly)
    else:
        lnglat_geom = wkt.loads(property_data)
        px_geom = polygon_to_pixel(lnglat_geom, bbox, image_shape)
        property_poly = geom_to_poly(px_geom)

    property_poly = property_poly.simplify(0.5, preserve_topology=True)
    property_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    property_mask, coord = add_geom_to_img(img=property_mask, geom=property_poly, color=(1,))
    return property_poly, property_mask, coord


import os

os.makedirs("test_parcel/train1", exist_ok=True)

gtu_ids = []
before_cleanup_len = []
after_cleanup_len = []
before_cleanup_coords = []
after_cleanup_coords = []
transformed_coords = []
before_img = []
ater_img = []
for i, record in enumerate(records):
    print(i)
    try:
        gtu_id, property_id = record.id, record.property_id
        gtu_id_dir, property_id_dir = possible_image_location(property_id=property_id, gtu_id=gtu_id)
        img1 = open_image(gtu_id_dir, property_id_dir, "image.jpg")

        try:
            if img1.shape[0] > 0:
                not_null = True
        except:
            not_null = False
        if not_null:
            if img1.shape[2] == 4:
                img1 = cv2.cvtColor(img1, cv2.COLOR_RGBA2RGB)
            img2 = img1.copy()
            before_img.append(img1)
            lnglat_coords = [(coord["lng"], coord["lat"]) for coord in record.boundary_data["coords"]]
            property_poly = coords_to_valid_poly(lnglat_coords)
            property_poly, property_mask1, coords = get_property_info(
                bbox=record.bbox,
                image_shape=img1.shape,
                property_data=property_poly.wkt,
                input_image_type_id=record.input_image_type_id,
                input_image_provider_type_id=record.input_image_provider_type_id,
            )
            coords1 = coords[0]
            coords2 = cleanup1(list(coords[0]))

            img2, coords3 = reshift_coords(img1, property_mask1, coords2, record.gsd)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"test_parcel/train/{gtu_id}.png", img2)
            gtu_ids.append(gtu_id)
            before_cleanup_len.append(len(coords1))
            before_cleanup_coords.append(coords1)
            after_cleanup_len.append(len(coords2))
            after_cleanup_coords.append(coords2)
            transformed_coords.append(coords3)
    except:
        print("hiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii biiiiiiiiiiiiiiiiiiiiiiiiiiatch ")


def sort_coords(coords):
    dst = list(map(distance, coords))
    origin = dst.index(min(dst))
    final_coords = coords[origin:] + coords[:origin]
    return final_coords


def interpolate(lol, n=20, t="same"):
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


def distance(l1, l2=[0, 0]):
    d = ((l1[0] - l2[0]) ** 2 + (l1[1] - l2[1]) ** 2) ** 0.5
    return d


df = pd.DataFrame()


df["gtu_ids"] = gtu_ids
df["before_cleanup_len"] = before_cleanup_len
df["after_cleanup_len"] = after_cleanup_len
df["before_cleanup_coords"] = before_cleanup_coords
df["after_cleanup_coords"] = after_cleanup_coords
df["transformed_coords"] = transformed_coords


df["before_cleanup_coords"] = df["before_cleanup_coords"].apply(lambda x: [list(i) for i in x])
df["after_cleanup_coords"] = df["after_cleanup_coords"].apply(lambda x: [list(i) for i in x])
df["coords_vals"] = df["transformed_coords"].apply(lambda x: [list(i) for i in x])


df["sorted_coords"] = df["coords_vals"].apply(sort_coords)
df["interpolate_same"] = df["sorted_coords"].apply(interpolate)
df["interpolate_same"] = df["interpolate_same"].apply(lambda x: [list(i) for i in x])
df["interpolate_linear"] = df["sorted_coords"].apply(lambda x: interpolate(x, t="linear"))
df["interpolate_linear"] = df["interpolate_linear"].apply(lambda x: [list(i) for i in x])


df.to_csv("dataset.csv", index=None)
