import cv2
import gcsfs
import numpy as np
import pandas as pd
from PIL import Image
from shapely import wkt
from bp.utils import add_geom_to_img
from shapely.geometry import Polygon
from bp.database import Gtu, db_session
from shapely.geometry import MultiPolygon
from  sqlalchemy.sql.expression import func
from bp.bounding_box import BoundingBox, polygon_to_pixel
from bp.enums import ImageryProviderTypes, InputImageTypes
from bp.boundary.utils import get_north_image_shape, get_oblique_poly, extend_property_poly
from bp.utils.helpers import polygon_coords_to_list, get_interiors_coords_list, geom_to_poly

FS = gcsfs.GCSFileSystem()

with db_session() as sess:
#     records = sess.query(Gtu).order_by(func.random()).limit(6400).all()
    records = sess.query(Gtu).offset(10000).limit(10).all()


def possible_image_location(property_id, gtu_id):
    """deterines the ideal image location (whether or not inside gtu folder)"""
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
    m = (y2-y1)/(x2-x1)
    return m

def distance(x1, y1, x2, y2):
    d = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
    return d

def overall_slope(coords):
    point=0
    while point < len(coords) - 1:
#         print(coords[point],coords[point+1])
        m = slope(coords[point][0],coords[point][1],coords[point+1][0],coords[point+1][1])
        d = distance(coords[point][0],coords[point][1],coords[point+1][0],coords[point+1][1])
#         print(d)
        if abs(m) < 0.2 and d < 500:
            coords.pop(point+1)
        else:
            point+=1
    return coords

def add_geom_to_img(
    img,
    geom,
    color = (1,),
    thickness = -1,
    modify_img = True):

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
    input_image_type_id = InputImageTypes.overhead(),
    input_image_provider_type_id = None,
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

    property_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    property_mask, coord = add_geom_to_img(img=property_mask, geom=property_poly, color=(1,))
    return property_poly, property_mask, coord