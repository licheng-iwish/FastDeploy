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
import os
import sys
import codecs
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
        print("postprocess input names:", self.input_names)

        self.output_names = []
        self.output_dtype = []
        for output_config in self.model_config["output"]:
            self.output_names.append(output_config["name"])
            dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])
            self.output_dtype.append(dtype)
        print("postprocess output names:", self.output_names)

        dir_name = os.path.dirname(os.path.realpath(__file__)) + "/"
        file_name = dir_name + "ppocr_keys_v1.txt"
        self.postprocessor = fd.vision.ocr.RecognizerPostprocessor(file_name)

    def filter(self, results, origin1, sens):
        ocr_rec_text = []
        ocr_rec_conf = []
        filter_origin1 = []
        for index, conf in enumerate(results[1]):
            if conf > 0.3 * (sens[1] + 1):
                ocr_rec_text.append(results[0][index])
                ocr_rec_conf.append(results[1][index])
                filter_origin1.append(origin1[index])
        return np.array(ocr_rec_text, dtype=np.object_), np.array(ocr_rec_conf), np.array(filter_origin1)

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
            rec_predict = pb_utils.get_input_tensor_by_name(request, self.input_names[0])
            rec_predict = rec_predict.as_numpy()
            det_area = pb_utils.get_input_tensor_by_name(request, self.input_names[1])
            det_area = det_area.as_numpy()
            labels = pb_utils.get_input_tensor_by_name(request, self.input_names[2])
            labels = labels.as_numpy()
            origin0 = pb_utils.get_input_tensor_by_name(request, self.input_names[3])
            origin0 = origin0.as_numpy()
            origin1 = pb_utils.get_input_tensor_by_name(request, self.input_names[4])
            origin1 = origin1.as_numpy()
            sens = pb_utils.get_input_tensor_by_name(request, self.input_names[5])
            sens = sens.as_numpy()
            results = self.postprocessor.run([rec_predict])
            ocr_rec_text, ocr_rec_conf, filter_origin1 = self.filter(results, origin1, sens)
            out_tensor_0 = pb_utils.Tensor(self.output_names[0], ocr_rec_text)
            out_tensor_1 = pb_utils.Tensor(self.output_names[1], ocr_rec_conf)
            out_tensor_2 = pb_utils.Tensor(self.output_names[2], det_area)
            out_tensor_3 = pb_utils.Tensor(self.output_names[3], labels)
            out_tensor_4 = pb_utils.Tensor(self.output_names[4], origin0)
            out_tensor_5 = pb_utils.Tensor(self.output_names[5], filter_origin1)
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0, out_tensor_1, out_tensor_2, out_tensor_3, out_tensor_4, out_tensor_5])
            responses.append(inference_response)
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
