#!/usr/bin/env python3
import cv2
import numpy as np
from data import load_data
import argparse


earth_radius = 6e3 # value doesn't actually matter


def geo_to_image_coords(longitude, latitude, image_width, image_height, earth_size_px):
    # if longitude < -180 or longitude >= 180:
    #     raise Exception(f"Invalid longitude {longitude}")
    # if latitude < -90 or latitude >= 90:
    #     raise Exception(f"Invalid latitude {latitude}")

    rx, ry, rz = geo_to_3D(longitude, latitude)
    if rx < 0:
        return None

    x_projected = ry
    y_projected = rz

    x = x_projected/(2*earth_radius) * earth_size_px + image_width/2
    y = -y_projected/(2*earth_radius) * earth_size_px + image_height/2
    # return (x, y)
    return (int(x), int(y))


def image_to_perspective(img, long, lat):
    width, height = img.shape[:2]

    pts_original_image = [(0, 0), (height, 0), (height, width), (0, width)]
    pts_original_xy = np.array([image_point_to_xy(xy, width, height) for xy in pts_original_image], dtype=np.float32)

    def rotx(angle):
        return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])

    def roty(angle):
        return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])

    pts_original_xy = np.hstack((pts_original_xy, np.zeros((4, 1))))
    pts_new_xy = (roty(np.deg2rad(long)) @ rotx(np.deg2rad(-lat)) @ pts_original_xy.T).T

    pts_new_image = [xy_to_image_point(xy, width, height) for xy in pts_new_xy[:, :2]]

    width_new = int(np.ceil(width*np.sqrt(2)))
    height_new = int(np.ceil(height*np.sqrt(2)))
    pts_new_image = pts_new_image + 0.5*np.ones((4,1)) @ (np.array([[width_new - width, height_new - height]]))

    pts_original_image = np.array(pts_original_image, dtype=np.float32)
    pts_new_image = np.array(pts_new_image, dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts_original_image, pts_new_image)
    new_img = cv2.warpPerspective(img, M, (width_new, height_new), flags=cv2.INTER_LINEAR)
    return new_img


def image_point_to_xy(image_point, width, height):
    x, y = image_point
    x, y = x - width//2, height//2 - y
    return (x, y)


def xy_to_image_point(xy, width, height):
    x, y = xy
    x, y = x + width//2, height//2 - y
    return (x, y)


def geo_to_3D(long, lat):
    long = np.deg2rad(long)
    lat = np.deg2rad(lat)

    x = earth_radius * np.cos(lat) * np.cos(long)
    y = earth_radius * np.cos(lat) * np.sin(long)
    z = earth_radius * np.sin(lat)

    return (x, y, z)


def overlay_image(full_img, overlay_img, alpha_img, xc, yc):
    """Inserts overlay_img into full_img at point xc, yc based on alpha map alpha_img"""
    assert full_img.dtype == np.float32
    assert overlay_img.dtype == np.float32
    assert alpha_img.dtype == np.float32

    # assert full_img.shape == (image_height, image_width, 3)
    image_height, image_width, _ = full_img.shape
    assert len(overlay_img.shape) == 3
    assert overlay_img.shape[2] == 3
    assert overlay_img.shape == alpha_img.shape

    x = xc - overlay_img.shape[0]//2
    y = yc - overlay_img.shape[1]//2

    # Image ranges
    y1, y2 = max(0, y), min(full_img.shape[0], y + overlay_img.shape[0])
    x1, x2 = max(0, x), min(full_img.shape[1], x + overlay_img.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(overlay_img.shape[0], full_img.shape[0] - y)
    x1o, x2o = max(0, -x), min(overlay_img.shape[1], full_img.shape[1] - x)

    alpha_img = alpha_img/255

    full_img_crop = full_img[y1:y2, x1:x2]
    overlay_img_crop = overlay_img[y1o:y2o, x1o:x2o]
    alpha_img_crop = alpha_img[y1o:y2o, x1o:x2o]

    full_img_crop[:] = full_img_crop*(1-alpha_img_crop) + overlay_img_crop*alpha_img_crop

    return full_img


def show_image(img):
    """Shows an image until you press 'X' to close it"""
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    window_name = "image"
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, 0, 0)
    while True:
        cv2.imshow(window_name, img)
        keyCode = cv2.waitKey(1)
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


def save_image(img, filepath):
    cv2.imwrite(filepath, img)


def generate_frame(angle, image_width, image_height, icon_size):
    code_to_info, edges = load_data()
    full_img = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    earth_size_px = min(image_width, image_height)*3/4

    np.random.seed(0)
    for i in range(1000):
        x = np.random.randint(0, full_img.shape[0])
        y = np.random.randint(0, full_img.shape[1])
        full_img[x, y] = (255, 255, 255)

    full_img = cv2.circle(full_img, (image_width//2, image_height//2), int(earth_size_px/2), (40, 10, 10), -1, lineType=cv2.LINE_AA)

    for (code1, code2) in edges:
        long = code_to_info[code1].longitude + angle
        lat = code_to_info[code1].latitude
        res = geo_to_image_coords(long, lat, image_width, image_height, earth_size_px)
        if res is not None:
            x1, y1 = res

            long = code_to_info[code2].longitude + angle
            lat = code_to_info[code2].latitude
            res = geo_to_image_coords(long, lat, image_width, image_height, earth_size_px)
            if res is not None:
                x2, y2 = res
                full_img = cv2.line(full_img, (x1, y1), (x2, y2), (255, 255, 255), 1, lineType=cv2.LINE_AA)

    full_img = full_img.astype(np.float32)
    circle = cv2.circle(np.zeros((icon_size, icon_size, 3), dtype=np.uint8), (icon_size//2, icon_size//2), icon_size//2, (255, 255, 255), -1, lineType=cv2.LINE_AA)
    for code in code_to_info:
        filename = code.lower()+".png"
        img = cv2.imread("h240/"+filename)
        img = cv2.resize(img, (icon_size, icon_size), cv2.INTER_LANCZOS4)

        long = code_to_info[code].longitude + angle
        lat = code_to_info[code].latitude
        res = geo_to_image_coords(long, lat, image_width, image_height, earth_size_px)
        if res is not None:
            x, y = res

            circle_perspective = image_to_perspective(circle, long, lat).astype(np.float32)
            img = image_to_perspective(img, long, lat).astype(np.float32)
            assert circle_perspective is not None
            full_img = overlay_image(full_img, img, circle_perspective, x, y)

    return full_img.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description='Generate an image of 3D globe.')
    parser.add_argument("width", help="width of output image", type=int)
    parser.add_argument("height", help="height of output image", type=int)
    args = parser.parse_args()

    image_width = args.width
    image_height = args.height
    icon_size = image_height//40
    angle = 0

    full_img = generate_frame(angle, image_width, image_height, icon_size)
    # show_image(full_img)
    save_image(full_img, f"media/image_{image_width}x{image_height}.png")


if __name__ == '__main__':
    main()
