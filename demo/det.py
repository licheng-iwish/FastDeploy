import cv2
import fastdeploy.vision as vision
from shapely.geometry import Polygon
import numpy as np
import math

def seal_text_expand(image, angle, dir):
    x0 = image.shape[0] // 2  # 计算圆心
    y0 = image.shape[1] // 2
    unwrapped_height = radius = min(x0, y0) # 通过圆形区域半径构造展开后的图像
    unwrapped_width = int(2 * math.pi * radius)
    unwrapped_img = np.zeros((unwrapped_height, unwrapped_width, 3), dtype="u1")
    pi = math.pi
    cos = math.cos
    sin = math.sin
    for n in range(unwrapped_width):
        #             3π/2
        #           π  -|-   0
        #              π/2
        theta = 2 * pi * (n / unwrapped_width) + angle  # 起始角度
        for m in range(unwrapped_height):
            unwrapped_radius = radius - m  # 2. don't forget
            if dir == 'clockwise':
                x = unwrapped_radius * sin(theta) + x0 - 1  # 3. "sin" is 顺时针 but "cos" is 逆时针
                y = unwrapped_radius * cos(theta) + y0 - 1  # 4. "sin" is 逆时针 but "cos" is 顺时针
            else:
                x = unwrapped_radius * cos(theta) + x0 - 1
                y = unwrapped_radius * sin(theta) + y0 - 1
            x, y = int(x), int(y)
            unwrapped_img[m, n, :] = image[x, y, :]
    if dir == 'clockwise':
        return moveBlack(unwrapped_img)
    else:
        return moveBlack(cv2.flip(unwrapped_img, 0))

def moveBlack(img):
    b = cv2.threshold(img, 3, 255, cv2.THRESH_BINARY)  # 调整裁剪效果
    binary_image = b[1]  # 二值图--具有三通道
    binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    edges_y, edges_x = np.where(binary_image == 255)  ##h, w
    bottom = min(edges_y)
    top = max(edges_y)
    left = min(edges_x)
    right = max(edges_x)

    height = top - bottom
    width = right - left

    res_image = img[bottom:bottom + height, left:left + width]
    return res_image

def preHandleImg(img, box):
    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    np_box = np.array(box)
    cv2.fillPoly(mask, [np_box], 255)
    # 复制图像
    img_cp = img.copy()
    img_cp[mask != 255, :] = 0
    return img_cp

def calcInclination(start, center):
    a = np.array([start[0] - center[0], start[1] - center[1]])
    b = np.array(center)
    cos_angle = a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.arccos(cos_angle)

def handleBox(img, box):
    mask_poly = Polygon(box)
    if mask_poly.area / mask_poly.minimum_rotated_rectangle.area < 0.9:  # 弯曲
        img = preHandleImg(img, box)
        angle = calcInclination(mask_poly.exterior.coords[0], [img.shape[1] / 2, img.shape[0] / 2])
        if mask_poly.centroid.y < img.shape[1] / 2:
            seal_img = seal_text_expand(img, angle, 'clockwise')
        else:
            seal_img = seal_text_expand(img, angle, 'anticlockwise')
    else:
        img = preHandleImg(img, box)
        seal_img = moveBlack(img)
    return seal_img

if __name__ == '__main__':
    im = cv2.imread("10.jpeg")
    # seal_det_model = vision.detection.PPYOLOE("inference/seal_det/model.pdmodel",
    #                                           "inference/seal_det/model.pdiparams",
    #                                           "inference/seal_det/infer_cfg.yml")

    ocr_det_model = vision.ocr.DBDetector("inference/ocr_det/inference.pdmodel",
                                          "inference/ocr_det/inference.pdiparams")

    ocr_det_model.preprocessor.max_side_len = 320
    ocr_det_model.postprocessor.det_db_use_ploy = True

    # ocr_rec_model = vision.ocr.Recognizer("inference/ocr_rec/inference.pdmodel",
    #                                       "inference/ocr_rec/inference.pdiparams",
    #                                       "inference/ocr_rec/ppocr_keys_v1.txt")

    # seal_det_res = seal_det_model.predict(im)
    # seal_det_imgs = []
    # for i, value in enumerate(seal_det_res.scores):
    #     if value > 0.8:
    #         box = seal_det_res.boxes[i]
    #         seal_det_imgs.append(im[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :])

    # for i, value in enumerate(seal_det_imgs):
    value = cv2.imread("box" + str(0) + ".jpg")
    c_value = value.copy()
    ocr_det_res = ocr_det_model.predict(c_value)
    print(ocr_det_res)


