
import cv2
import numpy as np

def get_approximate_square_crop_boxes(orig_shape, active_bbox):
    """
    Args:
        orig_shape:
        active_bbox (list): [min_x, max_x, min_y, max_y];

    Returns:

    """

    orig_h, orig_w = orig_shape

    min_x, min_y, max_x, max_y = active_bbox

    box_h = int(max_y - min_y)
    box_w = int(max_x - min_x)

    # print("orig = {}, active_bbox = {}, boxes = {}".format(orig_shape, active_bbox, (box_h, box_w)))
    if box_h > box_w:
        pad_size = box_h - box_w
        pad = pad_size // 2
        if pad_size % 2 == 0:
            pad_1, pad_2 = pad, pad
        else:
            pad_1, pad_2 = pad, pad + 1

        min_x = max(0, min_x - pad_1)
        max_x = min(orig_w, max_x + pad_2)

    else:
        pad_size = box_w - box_h
        pad = pad_size // 2
        if pad_size % 2 == 0:
            pad_1, pad_2 = pad, pad
        else:
            pad_1, pad_2 = pad, pad + 1

        min_y = max(0, min_y - pad_1)
        max_y = min(orig_h, max_y + pad_2)

    return min_x, min_y, max_x, max_y


def process_crop_img(orig_img, active_bbox, image_size, path=True):
    """
    Args:
        orig_img (np.ndarray):
        active_bbox (4,) : [x0, y0, x1, y1]
        image_size (int):

    Returns:
        image_info (dict): the information of processed image,
            `image` (np.ndarray): the crop and resized image, the shape is (image_size, image_size, 3);
            `orig_shape` (tuple): the shape of the original image;
            `active_bbox` (tuple or list): the active bbox [min_x, max_x, min_y, max_y];
            `factor`: the fact to enlarge the active bbox;
            `crop_bbox`: the cropped bbox [min_x, max_x, min_y, max_y];
            `pad_bbox`: the padded bbox [pad_left_x, pad_right_x, pad_top_y, pad_bottom_y],
    """
    if path:
        orig_img = cv2.imread(orig_img)
    x0, y0, x1, y1 = get_approximate_square_crop_boxes(orig_img.shape[0:2], active_bbox)
    crop_img = orig_img[y0: y1, x0: x1, :]
    crop_h, crop_w = crop_img.shape[0:2]

    start_pt = np.array([x0, y0], dtype=np.float32)

    pad_size = max(crop_h, crop_w) - min(crop_h, crop_w)
    pad = pad_size // 2
    if pad_size % 2 == 0:
        pad_1, pad_2 = pad, pad
    else:
        pad_1, pad_2 = pad, pad + 1

    # 1161 485 1080 1080 (1080, 1080, 3) (595, 0, 3) 595 297 298
    # print(x0, y0, x1, y1, orig_img.shape, crop_img.shape, pad_size, pad_1, pad_2)
    if crop_h < crop_w:
        crop_img = np.pad(
            array=crop_img,
            pad_width=((pad_1, pad_2), (0, 0), (0, 0)),
            mode="edge"
        )
        start_pt -= np.array([0, pad_1], dtype=np.float32)

    elif crop_h > crop_w:
        crop_img = np.pad(
            array=crop_img,
            pad_width=((0, 0), (pad_1, pad_2), (0, 0)),
            mode="edge"
        )
        start_pt -= np.array([pad_1, 0], dtype=np.float32)

    pad_crop_size = crop_img.shape[0]

    scale = image_size / pad_crop_size
    start_pt *= scale

    center = np.array([(x0 + x1) / 2, (y0 + y1) / 2], dtype=np.float32)
    center *= scale
    center -= start_pt

    proc_img = cv2.resize(crop_img, (image_size, image_size))

    return {
        "image": proc_img,
        "im_shape": orig_img.shape[0:2],
        "center": center,
        "scale": scale,
        "start_pt": start_pt,
    }


def update_active_boxes(cur_boxes, active_boxes=None):
    """

    Args:
        cur_boxes:
        active_boxes:

    Returns:

    """
    if active_boxes is None:
        active_boxes = cur_boxes
    else:
        active_boxes[0] = min(active_boxes[0], cur_boxes[0])
        active_boxes[1] = min(active_boxes[1], cur_boxes[1])
        active_boxes[2] = max(active_boxes[2], cur_boxes[2])
        active_boxes[3] = max(active_boxes[3], cur_boxes[3])

    return active_boxes


def fmt_active_boxes(active_boxes_XYXY, orig_shape, factor):
    boxes = enlarge_boxes(active_boxes_XYXY, orig_shape, factor)
    return pad_boxes(boxes, orig_shape)


def enlarge_boxes(active_boxes_XYXY,
                  orig_shape,
                  factor: float = 1.125):

    x0, y0, x1, y1 = active_boxes_XYXY
    height, width = orig_shape

    h = y1 - y0  # height

    ctr_x = (x0 + x1) // 2  # (center of x)
    ctr_y = (y0 + y1) // 2  # (center of y)

    _h = h * factor

    _y0 = max(0, int(ctr_y - _h / 2))
    _y1 = min(height, int(ctr_y + _h / 2))
    __h = _y1 - _y0

    _x0 = max(0, int(ctr_x - __h / 2))
    _x1 = min(width, int(ctr_x + __h / 2))

    return _x0, _y0, _x1, _y1


def pad_boxes(boxes_XYXY, orig_shape):
    orig_h, orig_w = orig_shape

    x0, y0, x1, y1 = boxes_XYXY

    box_h = int(x1 - x0)
    box_w = int(y1 - y0)

    if box_h > box_w:
        pad_size = box_h - box_w
        pad = pad_size // 2
        if pad_size % 2 == 0:
            pad_1, pad_2 = pad, pad
        else:
            pad_1, pad_2 = pad, pad + 1

        x0 = max(0, x0 - pad_1)
        x1 = min(orig_w, x1 + pad_2)

    else:
        pad_size = box_w - box_h
        pad = pad_size // 2
        if pad_size % 2 == 0:
            pad_1, pad_2 = pad, pad
        else:
            pad_1, pad_2 = pad, pad + 1

        y0 = max(0, y0 - pad_1)
        y1 = min(orig_h, y1 + pad_2)

    return x0, y0, x1, y1


def crop_resize_boxes(boxes, scale, start_pt):
    """
        crop and resize the boxes in the original image coordinates into cropped image coordinates.
    Args:
        boxes (list):
        scale:
        start_pt:

    Returns:
        new_boxes (list):
    """

    x0, y0, x1, y1 = np.copy(boxes)
    x0 = x0 * scale - start_pt[0]
    y0 = y0 * scale - start_pt[1]
    x1 = x1 * scale - start_pt[0]
    y1 = y1 * scale - start_pt[1]

    new_boxes = [x0, y0, x1, y1]

    return new_boxes


def crop_resize_kps(keypoints, scale, start_pt):
    """
        crop and resize the keypoints in the original image coordinates into cropped image coordinates.
    Args:
        keypoints (dict):
        scale:
        start_pt:

    Returns:
        new_kps (dict):
    """

    new_kps = dict()

    for part, kps in keypoints.items():
        if len(kps) > 0:
            renorm_kps = kps.copy()
            renorm_kps[:, 0:2] = renorm_kps[:, 0:2] * scale - start_pt[np.newaxis]
            new_kps[part] = renorm_kps
        else:
            new_kps[part] = kps

    return new_kps