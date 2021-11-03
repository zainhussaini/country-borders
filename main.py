import cv2
import numpy as np
from data import load_data
import argparse
from tqdm import tqdm
from multiprocessing import Pool
import os
import subprocess


EARTH_RADIUS = 6e3 # value doesn't actually matter
# Note: colors in (B, G, R) format (integer out of 255)
BACKGROUND_COLOR = (20, 20, 20)
EARTH_COLOR = (30, 30, 30)
BORDERS_COLOR = (255, 255, 255)
COUNTRY_SHADOW_COLOR = (60, 60, 60)
STARS_COLOR = (255, 255, 255)
CODEC = "mp4v"
VID_EXT = "mp4"


# this needs to be here because Windows does not support forking
CODE_TO_INFO, EDGES = load_data()
CODE_TO_IMGS = dict()
for code in CODE_TO_INFO:
    img = cv2.imread(os.path.join("h240", f"{code.lower()}.png"))
    # img = cv2.resize(img, (icon_size, icon_size), interpolation=cv2.INTER_AREA)
    assert img is not None
    CODE_TO_IMGS[code] = cv2.imread(os.path.join("h240", f"{code.lower()}.png"))


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


def calc_transform(img_shape, full_img_shape, icon_shape, long, lat):
    """Calculate transformation matrix from full image to icon_size with
    distortion based on long and lat.
    """
    # calculate pts_new_img as if you were taking a icon_size x icon_size
    # image and rotating about long and lat, then scaling the image to fit
    # around the new dimensions
    img_height, img_width = img_shape[:2]
    full_img_height, full_img_width = full_img_shape[:2]
    icon_height, icon_width = icon_shape[:2]

    earth_radius_px = min(full_img_width, full_img_height)*3/8

    angle_x = np.deg2rad(-lat)
    angle_y = np.deg2rad(long)

    # first step is to calculate where an icon that starts at centered in image ends up after
    # moving to longitude and latitude
    pts_original_xy = np.array([
        [-icon_width/2, icon_height/2, earth_radius_px],
        [icon_width/2, icon_height/2, earth_radius_px],
        [icon_width/2, -icon_height/2, earth_radius_px],
        [-icon_width/2, -icon_height/2, earth_radius_px]],
        dtype=np.float32)

    # these are the four points rotated about longitude latitude, and therefore
    # also shifted to where they should be on the globe in XY frame
    pts_new_xy = (roty(angle_y) @ rotx(angle_x) @ pts_original_xy.T).T

    z_avg_px = np.mean(pts_new_xy[:,-1])

    diag = np.diag(np.array([1, -1], np.float32))
    shift = np.ones((4, 1), np.float32) @ np.array([[full_img_width, full_img_height]], np.float32)/2
    pts_new_img = pts_new_xy[:, :2] @ diag + shift

    # calculate bounding box of pts_new_img
    ix, iy = np.floor(np.min(pts_new_img, axis=0))
    box_width, box_height = np.ceil(np.max(pts_new_img, axis=0)) - np.floor(np.min(pts_new_img, axis=0))

    # add a little padding around it
    padding = 4
    ix = ix - padding
    iy = iy - padding
    box_width = box_width + 2*padding
    box_height = box_height + 2*padding

    pts_new_img_centered = pts_new_img - np.ones((4, 1), np.float32) @ np.array([[ix, iy]], dtype=np.float32)
    assert (np.min(pts_new_img_centered, axis=0) >= 0).all()
    assert (np.max(pts_new_img_centered, axis=0) <= np.array([[box_width, box_height]])).all()

    # calculate the actual pts_original_img (make sure it lines up with pts_original_xy)
    pts_original_img = np.array([
        [0, 0],
        [img_width, 0],
        [img_width, img_height]],
        np.float32)

    M = cv2.getAffineTransform(pts_original_img[:3], pts_new_img_centered[:3])
    ix, iy, box_width, box_height = [int(i) for i in (ix, iy, box_width, box_height)]
    params = (M, ix, iy, box_width, box_height, z_avg_px)
    return params


def apply_transform(img, params, transform_img=True):
    assert img.dtype == np.uint8
    M, ix, iy, box_width, box_height, z_avg_px = params
    ellipse = cv2.ellipse(np.zeros(img.shape, np.uint8),
        (img.shape[1]//2, img.shape[0]//2),
        (img.shape[1]//2, img.shape[0]//2),
        0, 0, 360,
        (255, 255, 255),
        cv2.FILLED,
        cv2.LINE_8)

    upscale_factor = 4

    alpha_map = cv2.warpAffine(ellipse,
        upscale_factor*M,
        (upscale_factor*box_width, upscale_factor*box_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT)
    alpha_map = cv2.resize(alpha_map, (box_width, box_height), interpolation=cv2.INTER_AREA)

    # alpha_map = cv2.blur(alpha_map, (3, 3), borderType=cv2.BORDER_CONSTANT)
    alpha_map = alpha_map.astype(np.float32)/255

    if not transform_img:
        return alpha_map

    overlay_img = cv2.warpAffine(img,
        upscale_factor*M,
        (upscale_factor*box_width, upscale_factor*box_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REPLICATE)
    overlay_img = cv2.resize(overlay_img, (box_width, box_height), interpolation=cv2.INTER_AREA)
    return overlay_img, alpha_map


def overlay_image(full_img, overlay_img, alpha_map, x, y):
    """
    inputs:
        alpha_map - same shape as overlay_img but used to blend the overlay
        x, y - top left position of overlay_img in full_img
    """
    assert overlay_img.shape == alpha_map.shape
    assert full_img.dtype == np.uint8
    assert overlay_img.dtype == np.uint8
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

    full_img_cropped = full_img[y1:y2, x1:x2]
    alpha_map_cropped = alpha_map[y1o:y2o, x1o:x2o]
    overlay_img_cropped = overlay_img[y1o:y2o, x1o:x2o]
    # a = full_img_cropped
    # b = overlay_img_cropped
    # c = alpha_map_cropped*255
    # d = full_img_cropped*(1-alpha_map_cropped) + overlay_img_cropped*alpha_map_cropped
    # show_image(np.concatenate((np.concatenate((a, b), axis=1), np.concatenate((c, d), axis=1)), axis=0))
    full_img[y1:y2, x1:x2] = full_img_cropped*(1-alpha_map_cropped) + overlay_img_cropped*alpha_map_cropped


def show_image(img):
    """Shows an image until you press 'X' to close it"""
    if img.dtype != np.uint8:
        print("Warning: image not in np.uint8 format")
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


def draw_earth(full_img: np.ndarray):
    assert full_img.dtype == np.uint8

    # add circle for earth
    image_height, image_width, _ = full_img.shape
    earth_radius_px = min(image_width, image_height)*3/8
    xy_center = (image_width//2, image_height//2)

    shift = 2 # use this to get better accuracy radius
    full_img[:] = cv2.circle(full_img.copy(),
        center = [i * 2**shift for i in xy_center],
        radius = int(earth_radius_px * 2**shift),
        color = EARTH_COLOR,
        thickness = cv2.FILLED,
        lineType = cv2.LINE_8,
        shift = shift)


def draw_stars(full_img: np.ndarray):
    assert full_img.dtype == np.uint8

    image_height, image_width, _ = full_img.shape
    np.random.seed(0)
    for i in range(image_width*image_height//2048):
        ix = np.random.randint(0, image_width)
        iy = np.random.randint(0, image_height)

        full_img[iy, ix] = (255, 255, 255)
        # 5% of stars are extra big
        if i%20 == 0:
            if ix+1 < image_width and iy+1 < image_height:
                full_img[iy, ix+1] = (255, 255, 255)
                full_img[iy+1, ix] = (255, 255, 255)
                full_img[iy+1, ix+1] = (255, 255, 255)
            elif ix+1 < image_width:
                full_img[iy, ix+1] = (255, 255, 255)
            elif iy+1 < image_height:
                full_img[iy+1, ix] = (255, 255, 255)


def draw_borders(full_img, angle):
    assert full_img.dtype == np.uint8

    # draw shared border lines
    image_height, image_width, _ = full_img.shape
    earth_radius_px = min(image_width, image_height)*3/8
    lines = []
    for (code1, code2) in EDGES:
        long = CODE_TO_INFO[code1].longitude + angle
        lat = CODE_TO_INFO[code1].latitude
        x, y = geo_to_xy(long, lat, earth_radius_px)
        ix1, iy1 = xy_to_img(x, y, image_width, image_height)

        long = CODE_TO_INFO[code2].longitude + angle
        lat = CODE_TO_INFO[code2].latitude
        x, y = geo_to_xy(long, lat, earth_radius_px)
        ix2, iy2 = xy_to_img(x, y, image_width, image_height)

        lines.append([[ix1, iy1], [ix2, iy2]])

    # shift to get better accuracy on end point locations
    shift = 4
    full_img[:] = cv2.polylines(full_img,
        pts = (np.array(lines) * 2**shift).astype(np.int32),
        isClosed = False,
        color = BORDERS_COLOR,
        thickness = 1,
        lineType = cv2.LINE_AA,
        shift = shift)


def geo_to_xy(long: float, lat: float, earth_radius_px: float):
    # Convert geographic coordinates to position vector in 3D space (ECEF)
    long_rad = np.deg2rad(long)
    lat_rad = np.deg2rad(lat)
    rx = EARTH_RADIUS * np.cos(lat_rad) * np.cos(long_rad)
    ry = EARTH_RADIUS * np.cos(lat_rad) * np.sin(long_rad)
    rz = EARTH_RADIUS * np.sin(lat_rad)

    # convert to pixel coordinates
    x = ry*earth_radius_px/EARTH_RADIUS
    y = rz*earth_radius_px/EARTH_RADIUS

    return (x, y)


def xy_to_img(x: float, y: float, image_width: int, image_height: int):
    ix = x + image_width//2
    iy = image_height//2 - y
    return (ix, iy)


def img_to_xy(self, ix: float, iy: float, image_width: int, image_height: int):
    x = ix - image_width//2
    y = image_height//2  - iy
    return (x, y)


def draw_visible_flags(full_img: np.ndarray, angle):
    image_height, image_width, _ = full_img.shape
    icon_size = min(image_height, image_width)//40

    for code, img in CODE_TO_IMGS.items():
        long = CODE_TO_INFO[code].longitude + angle
        lat = CODE_TO_INFO[code].latitude

        params = calc_transform(img.shape, full_img.shape, (icon_size, icon_size), long, lat)
        z_avg_px = params[-1]

        if z_avg_px >= 0:
            overlay_img, alpha_map = apply_transform(img, params)
            overlay_image(full_img, overlay_img, alpha_map, params[1], params[2])


def draw_hidden_flags(full_img: np.ndarray, angle):
    image_height, image_width, _ = full_img.shape
    icon_size = min(image_height, image_width)//40

    for code, img in CODE_TO_IMGS.items():
        long = CODE_TO_INFO[code].longitude + angle
        lat = CODE_TO_INFO[code].latitude

        params = calc_transform(img.shape, full_img.shape, (icon_size, icon_size), long, lat)
        z_avg_px = params[-1]

        if z_avg_px < 0:
            alpha_map = apply_transform(img, params, False)
            overlay_img = np.zeros(alpha_map.shape, np.uint8)
            overlay_img[:] = COUNTRY_SHADOW_COLOR
            overlay_image(full_img, overlay_img, alpha_map, params[1], params[2])


def generate_frame(base_img, angle):
    full_img = base_img.copy()
    draw_hidden_flags(full_img, angle)
    draw_borders(full_img, angle)
    draw_visible_flags(full_img, angle)
    return full_img


def generate_image(base_img, angle=0):
    frame = generate_frame(base_img, angle)
    image_height, image_width, _ = frame.shape
    filepath = os.path.join("media", f"image_{image_width}x{image_height}.png")
    save_image(frame, filepath)


def helper(params):
    base_img, angle = params
    frame = generate_frame(base_img.copy(), angle)
    # return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def generate_video(base_img):
    image_height, image_width, _ = base_img.shape
    filepath = os.path.join(os.getcwd(), "media", f"video_{image_width}x{image_height}.{VID_EXT}")
    print(filepath)

    fps = 60
    time = 12
    angles = np.linspace(0, 360, fps*time, endpoint=False)

    params_list = ((base_img.copy(), angle) for angle in angles)

    fourcc = cv2.VideoWriter.fourcc(*CODEC)
    writer = cv2.VideoWriter(filepath, fourcc, fps, (image_width, image_height))
    with Pool() as pool:
        for frame in tqdm(pool.imap(helper, params_list), total=len(angles)):
            writer.write(frame)
    writer.release()

    temppath = "temp.mp4"
    subprocess.run(f"mv {filepath} {temppath}".split(" "))
    # https://gist.github.com/Vestride/278e13915894821e1d6f
    subprocess.run(f"ffmpeg -an -i {temppath} -vcodec libx264 -pix_fmt yuv420p -profile:v baseline -level 3 {filepath}".split(" "))
    subprocess.run(f"rm {temppath}".split(" "))

    # subprocess.run(f"mpv {filepath} --fs --loop".split(" "))


def generate_video_sequential(base_img):
    image_height, image_width, _ = base_img.shape
    filepath = os.path.join("media", f"video_{image_width}x{image_height}.{VID_EXT}")

    fps = 1
    time = 12
    angles = np.linspace(0, 360, int(fps*time), endpoint=False)

    fourcc = cv2.VideoWriter.fourcc(*CODEC)
    writer = cv2.VideoWriter(filepath, fourcc, fps, (image_width, image_height))
    for angle in tqdm(angles):
        params = (base_img, angle)
        frame = helper(params)
        show_image(frame)
        assert frame.shape == (image_height, image_width, 3)
        writer.write(frame)
    writer.release()

    # temppath = "temp.mp4"
    # subprocess.run(f"mv {filepath} {temppath}".split(" "))
    # subprocess.run(f"ffmpeg -an -i {temppath} -vcodec libx264 -pix_fmt yuv420p -profile:v baseline -level 3 {filepath}".split(" "))
    # subprocess.run(f"rm {temppath}".split(" "))

    # subprocess.run(f"mpv {filepath} --fs --loop".split(" "))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate an image of 3D globe.")
    parser.add_argument("width", help="width of output image", type=int, nargs="?", const=1920, default=1920)
    parser.add_argument("height", help="height of output image", type=int, nargs="?", const=1080, default=1080)
    args = parser.parse_args()
    image_width = args.width
    image_height = args.height

    angle = 0

    set1 = set(CODE_TO_INFO.keys())
    set2 = set(CODE_TO_IMGS.keys())
    assert set1 == set2

    base_img = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    base_img[:,:] = BACKGROUND_COLOR
    draw_stars(base_img)
    draw_earth(base_img)

    # generate_image(base_img, angle)
    # generate_video_sequential(base_img)
    generate_video(base_img)
