#!/usr/bin/env python3
import cv2
import numpy as np
from data import load_data
import argparse
from tqdm import tqdm


EARTH_RADIUS = 6e3 # value doesn't actually matter
BACKGROUND_COLOR = (20, 20, 20) # BGR
EARTH_COLOR = (30, 30, 30) # BGR
BORDERS_COLOR = (255, 255, 255) # BGR
COUNTRY_SHADOW_COLOR = (60, 60, 60)


def show_image(img):
    """Shows an image until you press 'X' to close it"""
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    window_name = "image"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    height, width, _ = img.shape
    if height > 1000 or width > 1920:
        new_width = 1920//4*3
        new_height = 1080//4*3
    elif min(height, width) < 600:
        new_width = 600
        new_height = 600
    else:
        new_width = width
        new_height = height

    cv2.resizeWindow('image', new_width, new_height)
    cv2.moveWindow(window_name, (1920-new_width)//2, (1080-new_height)//2)

    while True:
        cv2.imshow(window_name, img)
        keyCode = cv2.waitKey(1)
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


def save_image(img, filepath):
    cv2.imwrite(filepath, img)


def overlay_image(full_img, overlay_img, alpha_map, x, y):
    """
    inputs:
        alpha_map - same shape as overlay_img but used to blend the overlay
        x, y - top left position of overlay_img in full_img
    """
    assert overlay_img.shape == alpha_map.shape
    assert full_img.dtype == np.float32
    assert overlay_img.dtype == np.float32
    assert alpha_map.dtype == np.float32
    assert np.max(alpha_map) <= 1 and np.min(alpha_map) >= 0

    full_img_height, full_img_width, _ = full_img.shape
    overlay_img_height, overlay_img_width, _ = overlay_img.shape

    # Image ranges
    y1, y2 = max(0, y), min(full_img_height, y + overlay_img_height)
    x1, x2 = max(0, x), min(full_img_width, x + overlay_img_width)

    # Overlay ranges
    y1o, y2o = max(0, -y), min(overlay_img_height, full_img_height - y)
    x1o, x2o = max(0, -x), min(overlay_img_width, full_img_width - x)

    full_img[y1:y2, x1:x2] = full_img[y1:y2, x1:x2]*(1-alpha_map) + overlay_img[y1o:y2o, x1o:x2o]*alpha_map

def rotx(angle):
        return np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]],
            dtype=np.float32)


def roty(angle):
    return np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]],
        dtype=np.float32)


class ImageGenerator:
    def __init__(self, image_width, image_height):
        if image_width <= 40 or image_height <= 40:
            raise Exception("ImageGenerator image dimensions need to be greater than 40x40")
        self.image_width = image_width
        self.image_height = image_height

        self.icon_size = image_height//40
        self.earth_radius_px = min(image_width, image_height)*3/8

        # generate background image
        full_img_base = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        full_img_base[:,:] = BACKGROUND_COLOR

        # add stars
        np.random.seed(0)
        for i in range(image_width*image_height//2048):
            ix = np.random.randint(0, image_width)
            iy = np.random.randint(0, image_height)
            full_img_base[iy, ix] = (255, 255, 255)

        self.full_img_base = full_img_base

        self.code_to_info, self.edges = load_data()
        self.code_to_img = dict()
        for code in self.code_to_info:
            filename = code.lower()+".png"
            self.code_to_img[code] = cv2.imread("h240/"+filename)

    def generate(self, angle=0):
        z_poss, params = self.get_flag_params(angle)
        z_poss_params = list(zip(z_poss, params))
        behind_params = [params for z_poss, params in z_poss_params if z_poss <= 0]
        front_params = [params for z_poss, params in z_poss_params if z_poss > 0]

        full_img = self.full_img_base

        full_img = self.draw_earth(full_img)
        full_img = self.draw_country_flags(full_img, behind_params, darken=True)
        full_img = self.draw_borders(full_img, angle)
        full_img = self.draw_country_flags(full_img, front_params, darken=False)

        return full_img

    def draw_earth(self, full_img):
        # add circle for earth
        xy_center = (self.image_width//2, self.image_height//2)
        shift = 2 # use this to get better accuracy radius
        full_img = cv2.circle(full_img.copy(),
            center = [i * 2**shift for i in xy_center],
            radius = int(self.earth_radius_px * 2**shift),
            color = EARTH_COLOR,
            thickness = cv2.FILLED,
            lineType = cv2.LINE_AA,
            shift = shift)
        return full_img

    def draw_borders(self, full_img, angle):
        # draw shared border lines
        lines = []
        for (code1, code2) in self.edges:
            long = self.code_to_info[code1].longitude + angle
            lat = self.code_to_info[code1].latitude
            ix1, iy1 = self.xy_to_img(*self.geo_to_xy(long, lat))

            long = self.code_to_info[code2].longitude + angle
            lat = self.code_to_info[code2].latitude
            ix2, iy2 = self.xy_to_img(*self.geo_to_xy(long, lat))

            lines.append([[ix1, iy1], [ix2, iy2]])

        shift = 2
        full_img = cv2.polylines(full_img,
            pts = (np.array(lines) * 2**shift).astype(np.int32),
            isClosed = False,
            color = BORDERS_COLOR,
            thickness = 1,
            lineType = cv2.LINE_AA,
            shift = shift)

        return full_img

    def draw_country_flags(self, full_img, params, darken=False):
        # draw country flags
        full_img = full_img.astype(np.float32)

        circle = cv2.circle(np.zeros((self.icon_size, self.icon_size, 3), dtype=np.uint8),
            center = (self.icon_size//2, self.icon_size//2),
            radius = self.icon_size//2,
            color = (255, 255, 255),
            thickness = cv2.FILLED,
            lineType = cv2.LINE_AA)

        for (code, M, ix, iy, box_width, box_height) in params:
            img = self.code_to_img[code]

            alpha_map = cv2.warpPerspective(circle,
                M = M,
                dsize = (box_width, box_height),
                flags = cv2.INTER_LINEAR,
                borderMode = cv2.BORDER_CONSTANT,
                borderValue = 0).astype(np.float32)/255

            if darken:
                new_img = np.zeros((box_height, box_width, 3), np.float32)
                new_img[:, :] = COUNTRY_SHADOW_COLOR
            else:
                # M, ix, iy, box_width, box_height = self.calc_transform(img.shape, long, lat)
                # new_img = cv2.warpPerspective(img,
                #     M = M,
                #     dsize = (box_width, box_height),
                #     flags = cv2.INTER_LINEAR,
                #     cv2.BORDER_REPLICATE).astype(np.float32)

                new_img = cv2.resize(img,
                    (self.icon_size, self.icon_size),
                    cv2.INTER_LANCZOS4)

                new_img = cv2.warpPerspective(new_img,
                    M = M,
                    dsize = (box_width, box_height),
                    flags = cv2.INTER_LINEAR,
                    borderMode = cv2.BORDER_REPLICATE).astype(np.float32)

            overlay_image(full_img, new_img, alpha_map, ix, iy)

        return full_img.astype(np.uint8)

    def get_flag_params(self, angle):
        # Perform all the computation here so you can insert flags in correct order
        z_poss = []
        params = []
        for code, img in self.code_to_img.items():
            long = self.code_to_info[code].longitude + angle
            lat = self.code_to_info[code].latitude

            M, ix, iy, box_width, box_height, z_pos = self.calc_transform(
                (self.icon_size, self.icon_size), long, lat)
            z_poss.append(z_pos)
            params.append((code, M, ix, iy, box_width, box_height))

        return z_poss, params

    def geo_to_xy(self, long, lat):
        # Convert geographic coordinates to position vector in 3D space (ECEF)
        long_rad = np.deg2rad(long)
        lat_rad = np.deg2rad(lat)
        rx = EARTH_RADIUS * np.cos(lat_rad) * np.cos(long_rad)
        ry = EARTH_RADIUS * np.cos(lat_rad) * np.sin(long_rad)
        rz = EARTH_RADIUS * np.sin(lat_rad)

        # convert to pixel coordinates
        x = ry*self.earth_radius_px/EARTH_RADIUS
        y = rz*self.earth_radius_px/EARTH_RADIUS

        return (x, y)

    def xy_to_img(self, x, y):
        ix = x + self.image_width//2
        iy = self.image_height//2 - y
        return (ix, iy)

    def img_to_xy(self, ix, iy):
        x = ix - self.image_width//2
        y = self.image_height//2  - iy
        return (x, y)

    def calc_transform(self, shape, long, lat):
        """Calculate transformation matrix from full image to icon_size with
        distortion based on long and lat.
        """
        # calculate pts_new_img as if you were taking a icon_size x icon_size
        # image and rotating about long and lat, then scaling the image to fit
        # around the new dimensions
        image_width = self.icon_size
        image_height = self.icon_size

        angle_x = np.deg2rad(-lat)
        angle_y = np.deg2rad(long)

        pts_original_img = np.array([
            (0, 0),
            (image_width, 0),
            (image_width, image_height),
            (0, image_height)], np.float32)

        diag = np.diag(np.array([1, -1], np.float32))
        four_ones = np.ones((4, 1), np.float32)

        shift = np.array([[image_width, image_height]], np.float32)/2
        pts_original_xy = (pts_original_img - four_ones @ shift) @ diag
        assert (np.min(pts_original_xy, axis=0) >= -shift).all()
        assert (np.max(pts_original_xy, axis=0) <= shift).all()

        # these are four points in X, Y frame (with Z equal to earth radius)
        pts_original_xy = np.hstack((
            pts_original_xy, self.earth_radius_px * np.ones((4, 1), np.float32)))

        # these are the four points rotated about longitude latitude, and therefore
        # also shifted to where they should be on the globe in X, Y frame still
        pts_new_xy = (roty(angle_y) @ rotx(angle_x) @ pts_original_xy.T).T

        z_pos = np.mean(pts_new_xy, axis=0)[-1]
        pts_original_xy = pts_original_xy[:, :2]
        pts_new_xy = pts_new_xy[:, :2]

        image_width_new, image_height_new = self.image_width, self.image_height
        shift = np.array([[image_width_new, image_height_new]], np.float32)/2
        pts_new_img = pts_new_xy @ diag + four_ones @ shift

        assert (np.min(pts_new_img, axis=0) >= 0).all()
        assert (np.max(pts_new_img, axis=0) <= np.array([[image_width_new, image_height_new]])).all()

        # calculate bounding box of pts_new_img
        ix, iy = np.floor(np.min(pts_new_img, axis=0))
        box_width, box_height = np.ceil(np.max(pts_new_img, axis=0)) - np.floor(np.min(pts_new_img, axis=0))

        pts_new_img_centered = pts_new_img - four_ones @ np.array([[ix, iy]])
        assert (np.min(pts_new_img_centered, axis=0) >= 0).all()
        assert (np.max(pts_new_img_centered, axis=0) <= np.array([[box_width, box_height]])).all()

        # calculate the actual pts_original_img
        image_height, image_width = shape
        pts_original_img = np.array([
            (0, 0),
            (image_width, 0),
            (image_width, image_height),
            (0, image_height)], np.float32)

        M = cv2.getPerspectiveTransform(pts_original_img, pts_new_img_centered)
        ix, iy, box_width, box_height = [int(i) for i in (ix, iy, box_width, box_height)]
        return M, ix, iy, box_width, box_height, z_pos


def main():
    parser = argparse.ArgumentParser(description='Generate an image of 3D globe.')
    parser.add_argument("width", help="width of output image", type=int)
    parser.add_argument("height", help="height of output image", type=int)
    args = parser.parse_args()

    image_width = args.width
    image_height = args.height
    angle = 0

    ig = ImageGenerator(image_width, image_height)
    frame = ig.generate(angle)
    save_image(frame, f"media/image_{image_width}x{image_height}.png")
    # show_image(frame)


if __name__ == '__main__':
    main()
