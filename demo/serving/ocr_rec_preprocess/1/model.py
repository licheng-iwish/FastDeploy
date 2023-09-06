# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import numpy as np
import time
import cv2
from shapely.geometry import Polygon
import math
import fastdeploy as fd

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        # You must parse model_config. JSON string is not parsed here
        self.model_config = json.loads(args['model_config'])
        print("model_config:", self.model_config)

        self.input_names = []
        for input_config in self.model_config["input"]:
            self.input_names.append(input_config["name"])
        print("preprocess input names:", self.input_names)

        self.output_names = []
        self.output_dtype = []
        for output_config in self.model_config["output"]:
            self.output_names.append(output_config["name"])
            dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])
            self.output_dtype.append(dtype)
        print("preprocess output names:", self.output_names)
        self.preprocessor = fd.vision.ocr.RecognizerPreprocessor()

    def get_seal_imgs(self, origin0, row_imgs, boxes):
        seal_imgs = []
        for i in range(origin0.shape[0]):
            row_img = row_imgs[origin0[i]]
            box = boxes[i]
            box_row_img = row_img[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
            seal_imgs.append(box_row_img)
        return seal_imgs

    def polar_coordinates_expand(self, img, angle, clock_dir):
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
                radius = int((x0 * y0) / (((x0 ** 2) * (math.cos(theta) ** 2) + (y0 ** 2) * (
                    math.sin(theta) ** 2)) ** 0.5))
                unwrapped_radius = radius - i  # 2. don't forget
                if clock_dir == 'clockwise':
                    x = unwrapped_radius * sin(theta) + x0 - 1  # 3. "sin" is 顺时针 but "cos" is 逆时针
                    y = unwrapped_radius * cos(theta) + y0 - 1  # 4. "sin" is 逆时针 but "cos" is 顺时针
                else:
                    x = unwrapped_radius * cos(theta) + x0 - 1
                    y = unwrapped_radius * sin(theta) + y0 - 1
                unwrapped_img[i, j, :] = img[int(x), int(y), :]
        if clock_dir == 'clockwise':
            return self.move_black(unwrapped_img)
        else:
            return self.move_black(cv2.flip(unwrapped_img, 0))

    def move_black(self, img):
        edges_y, edges_x, _ = np.where(img != 0)  ##h, w
        bottom = min(edges_y)
        top = max(edges_y)
        left = min(edges_x)
        right = max(edges_x)
        height = top - bottom
        width = right - left
        res_img = img[bottom:bottom + height, left:left + width]
        return res_img

    def intercept_rec_area(self, seal_img, det_boxes):
        mask = np.zeros((seal_img.shape[0], seal_img.shape[1]), np.uint8)
        np_boxes = np.array(det_boxes)
        cv2.fillPoly(mask, [np_boxes], 255)
        # 复制图像
        img_cp = seal_img.copy()
        img_cp[mask!=255, :] = 0
        return img_cp

    def check_curvature(self, seal_img, det_boxes):
        mask_poly = Polygon(det_boxes)
        if mask_poly.area / mask_poly.minimum_rotated_rectangle.area < 0.6:  # 弯曲
            start = min(mask_poly.exterior.coords)
            center = [seal_img.shape[1] / 2, seal_img.shape[0] / 2]
            a = np.array([start[0] - center[0], start[1] - center[1]])
            b = np.array(center)
            cos_angle = a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))
            angle = np.arccos(cos_angle)
            if mask_poly.centroid.y < center[0]:
                return angle, "clockwise"
            else:
                return angle, "anticlockwise"    
        return None, None

    def get_seal_rec_imgs(self, origin1, labels, seal_imgs, ocr_det_boxes, ocr_det_boxes_len):
        seal_rec_imgs = []    
        ocr_det_boxes_len_start = 0
        for i in range(origin1.shape[0]):
            seal_img = seal_imgs[origin1[i]]
            box = ocr_det_boxes[ocr_det_boxes_len_start : ocr_det_boxes_len_start + ocr_det_boxes_len[i]]
            ocr_det_boxes_len_start += ocr_det_boxes_len[i]
            rec_area_img = self.intercept_rec_area(seal_img, box)
            if labels[origin1[i]] == 1 or labels[origin1[i]] == 3:
                rec_area_img = self.move_black(rec_area_img)
                seal_rec_imgs.append(rec_area_img.copy())
            else:
                angle, clock_dir = self.check_curvature(seal_img, box)
                if angle == None:
                    rec_area_img = self.move_black(rec_area_img)
                    seal_rec_imgs.append(rec_area_img.copy())
                else:
                    try: 
                        rec_area_img = self.polar_coordinates_expand(rec_area_img, angle, clock_dir)
                    except:
                        print("polar_coordinates_expand error")
                    else:
                        seal_rec_imgs.append(rec_area_img.copy())               
        return seal_rec_imgs    

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        responses = []
        for request in requests:
            row_imgs = pb_utils.get_input_tensor_by_name(request,
                                                     self.input_names[0])
            row_imgs = row_imgs.as_numpy()
            boxes = pb_utils.get_input_tensor_by_name(request,
                                                     self.input_names[1])
            boxes = boxes.as_numpy()
            labels = pb_utils.get_input_tensor_by_name(request,
                                                     self.input_names[2])
            labels = labels.as_numpy()
            origin0 = pb_utils.get_input_tensor_by_name(request,
                                                     self.input_names[3])
            origin0 = origin0.as_numpy()
            origin1 = pb_utils.get_input_tensor_by_name(request,
                                                     self.input_names[4])
            origin1 = origin1.as_numpy()
            ocr_det_boxes =  pb_utils.get_input_tensor_by_name(request,
                                                     self.input_names[5])     
            ocr_det_boxes = ocr_det_boxes.as_numpy()
            ocr_det_boxes_len = pb_utils.get_input_tensor_by_name(request,
                                                     self.input_names[6])
            ocr_det_boxes_len = ocr_det_boxes_len.as_numpy()                                              
            seal_imgs = self.get_seal_imgs(origin0, row_imgs, boxes)
            seal_rec_imgs = self.get_seal_rec_imgs(origin1, labels, seal_imgs, ocr_det_boxes, ocr_det_boxes_len)
            outputs = self.preprocessor.run(seal_rec_imgs)
            dlpack_tensor = outputs[0].to_dlpack()
            output_tensor = pb_utils.Tensor.from_dlpack(self.output_names[0],
                                                          dlpack_tensor)
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor, ])
            responses.append(inference_response)
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
