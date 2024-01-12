import cv2
import fastdeploy.vision as vision
import numpy
from shapely import LineString, Point
from shapely.geometry import Polygon
import numpy as np
import math


def polar_coordinates_expand(img, angle):
    x0 = img.shape[0] // 2  # 计算圆心
    y0 = img.shape[1] // 2
    unwrapped_height = img.shape[0] // 2  # 通过圆形区域半径构造展开后的图像
    unwrapped_width = int(math.pi * y0)
    unwrapped_img = np.zeros((unwrapped_height, unwrapped_width, 3), dtype=np.uint8)
    pi = math.pi
    cos = math.cos
    sin = math.sin
    for j in range(unwrapped_width):
        #             3π/2
        #           π  -|-   0
        #              π/2
        theta = 2 * pi * (j / unwrapped_width) + angle  # 起始角度
        for i in range(unwrapped_height):
            # x = unwrapped_radius * sin(theta) + x0 - 1  # 3. "sin" is 顺时针 but "cos" is 逆时针
            # y = unwrapped_radius * cos(theta) + y0 - 1  # 4. "sin" is 逆时针 but "cos" is 顺时针
            radius = int((x0 * y0) / (((x0 ** 2) * (math.cos(theta) ** 2) + (y0 ** 2) * (
                    math.sin(theta) ** 2)) ** 0.5))
            unwrapped_radius = radius - i  # 2. don't forget
            x = unwrapped_radius * sin(theta) + x0 - 1  # 3. "sin" is 顺时针 but "cos" is 逆时针
            y = unwrapped_radius * cos(theta) + y0 - 1  # 4. "sin" is 逆时针 but "cos" is 顺时针
            if int(x) >= img.shape[0] or int(y) >= img.shape[1]:
                unwrapped_img[i, j, :] = 0
            else:
                unwrapped_img[i, j, :] = img[int(x), int(y), :]
    return move_black(unwrapped_img)


def move_black(img):
    edges_y, edges_x, _ = np.where(img != 0)  ##h, w
    bottom = min(edges_y)
    top = max(edges_y)
    left = min(edges_x)
    right = max(edges_x)
    height = top - bottom
    width = right - left
    res_img = img[bottom:bottom + height, left:left + width]
    return res_img


def intercept_rec_area(seal_img, det_boxes):
    mask = np.zeros((seal_img.shape[0], seal_img.shape[1]), np.uint8)
    np_boxes = np.array(det_boxes)
    cv2.fillPoly(mask, [np_boxes], 255)
    # 复制图像
    img_cp = seal_img.copy()
    img_cp[mask != 255, :] = 0
    return img_cp


def check_curvature(seal_img, det_boxes):
    mask_poly = Polygon(det_boxes)
    if mask_poly.area / mask_poly.minimum_rotated_rectangle.area < 0.6:  # 弯曲
        mrrbc = mask_poly.minimum_rotated_rectangle.boundary.coords
        center = [seal_img.shape[1] / 2, seal_img.shape[0] / 2]
        for i in range(len(mrrbc) - 1):
            line = LineString([((mrrbc[i + 1][0] + mrrbc[i][0]) / 2, (mrrbc[i + 1][1] + mrrbc[i][1]) / 2),
                               (center[0], center[1])])
            if not line.intersects(mask_poly):
                break
        start = [(mrrbc[i + 1][0] + mrrbc[i][0]) / 2, (mrrbc[i + 1][1] + mrrbc[i][1]) / 2]
        a = np.array([start[0] - center[0], center[1] - start[1]])
        b = np.array([1, 0])
        cos_angle = a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))
        angle = np.arccos(cos_angle)
        if start[1] < center[1]:
            angle = 2 * math.pi - angle
        if not Point(center).within(mask_poly.minimum_rotated_rectangle):
            angle = math.pi + angle
        return angle
    return None

def matrix_leveling(src, angle):
    rows, cols, _ = src.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1.0)
    warpAffine_img = cv2.warpAffine(src, M=M, dsize=(cols, rows))
    return warpAffine_img

def get_seal_rec_imgs(seal_img, boxes):
    seal_rec_imgs = []
    for box in boxes:
        rec_area_img = intercept_rec_area(seal_img, box)
        angle = check_curvature(seal_img, box)
        if angle == None:
            rect = cv2.minAreaRect(numpy.asarray(box))
            rec_area_img = matrix_leveling(rec_area_img, rect[2])
            rec_area_img = move_black(rec_area_img)
            seal_rec_imgs.append(rec_area_img)
        else:
            try:
                rec_area_img = polar_coordinates_expand(rec_area_img, angle)
            except:
                print("polar_coordinates_expand error")
            else:
                seal_rec_imgs.append(rec_area_img)
    return seal_rec_imgs

def draw_det_res(dt_boxes, img):
    if len(dt_boxes) > 0:
        src_im = img
        for box in dt_boxes:
            box = np.array(box).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
            #draw_det_res_rang(box, src_im)
        cv2.imwrite("test1.png", src_im)


def draw_det_res_rang(box, img):
    rect = cv2.minAreaRect(box)
    # cv2.boxPoints可以将轮廓点转换为四个角点坐标
    boxEx = cv2.boxPoints(rect)
    # 在原图上画出预测的外接矩形
    boxEx = boxEx.reshape((-1,1,2)).astype(np.int32)
    cv2.polylines(img, [boxEx], True, color=(0, 255, 0), thickness=2)

if __name__ == '__main__':
    im = cv2.imread("2.png")
    # seal_det_model = vision.detection.PPYOLOE("inference/seal_det/model.pdmodel",
    #                                           "inference/seal_det/model.pdiparams",
    #                                           "inference/seal_det/infer_cfg.yml")

    ocr_det_model = vision.ocr.DBDetector("inference/ocr_det/inference.pdmodel",
                                          "inference/ocr_det/inference.pdiparams")

    ocr_det_model.preprocessor.max_side_len = 320
    ocr_det_model.postprocessor.det_db_unclip_ratio = 3.5
    ocr_det_model.postprocessor.det_db_use_ploy = True

    ocr_rec_model = vision.ocr.Recognizer("inference/ocr_rec/inference.pdmodel",
                                          "inference/ocr_rec/inference.pdiparams",
                                          "inference/ocr_rec/ppocr_keys_v1.txt")

    # seal_det_res = seal_det_model.predict(im)
    # seal_det_imgs = []
    # for i, value in enumerate(seal_det_res.scores):
    #     if value > 0.8:
    #         box = seal_det_res.boxes[i]
    #         seal_det_imgs.append(im[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :])

    # for i, value in enumerate(seal_det_imgs):
    # value = cv2.imread("box" + str(0) + ".jpg")
    c_value = im.copy()
    ocr_det_res = ocr_det_model.predict(c_value)
    boxes = ocr_det_res.boxes[0:3]
    #draw_det_res(boxes, c_value)
    rets = get_seal_rec_imgs(c_value, boxes)
    for i, box in enumerate(rets):
        cv2.imwrite(str(i)+"2.png", box)
        if i == 0:
            box = cv2.rotate(box, rotateCode=cv2.ROTATE_180)
        print(ocr_rec_model.predict(box.copy()))
