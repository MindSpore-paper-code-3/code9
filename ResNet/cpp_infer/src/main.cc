/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <sys/time.h>
#include <dirent.h>
#include <iostream>
#include <cstring>
#include <string>
#include <algorithm>
#include <iosfwd>
#include <vector>
#include <fstream>
#include <sstream>

#include "common_inc/infer.h"

DEFINE_string(mindir_path, "", "mindir path");
DEFINE_string(dataset_path, ".", "dataset path");
DEFINE_string(network, "resnet18", "networktype");
DEFINE_string(dataset, "imagenet", "dataset");
DEFINE_string(device_type, "CPU", "device type");
DEFINE_int32(device_id, 0, "device id");

int main(int argc, char **argv) {
  if (!ParseCommandLineFlags(argc, argv)) {
    std::cout << "Failed to parse args" << std::endl;
    return 1;
  }
  if (RealPath(FLAGS_mindir_path).empty()) {
    std::cout << "Invalid mindir path: " << FLAGS_mindir_path << std::endl;
    return 1;
  }
  Status ret;
  Model model;
  if (!LoadModel(FLAGS_mindir_path, FLAGS_device_type, FLAGS_device_id, &model)) {
    std::cout << "Failed to load model " << FLAGS_mindir_path << ", device id: " << FLAGS_device_id
              << ", device type: " << FLAGS_device_type;
    return 1;
  }

  auto all_files = GetAllFiles(FLAGS_dataset_path);
  if (all_files.empty()) {
    std::cout << "ERROR: no input data." << std::endl;
    return 1;
  }

  std::vector<MSTensor> modelInputs = model.GetInputs();
  std::map<double, double> costTime_map;
  size_t size = all_files.size();

  std::shared_ptr<TensorTransform> decode = std::make_shared<Decode>();
  std::shared_ptr<TensorTransform> hwc2chw = std::make_shared<HWC2CHW>();

  std::shared_ptr<TensorTransform> resize = std::make_shared<Resize>(std::vector<int>{256});
  std::shared_ptr<TensorTransform> centercrop = std::make_shared<CenterCrop>(std::vector<int>{224});
  std::shared_ptr<TensorTransform> normalize =
    std::make_shared<Normalize>(std::vector<float>{123.675, 116.28, 103.53}, std::vector<float>{58.395, 57.12, 57.375});

  std::shared_ptr<TensorTransform> normalizeResnet101 =
    std::make_shared<Normalize>(std::vector<float>{121.125, 115.005, 99.96}, std::vector<float>{70.125, 68.085, 70.89});

  std::shared_ptr<TensorTransform> sr_resize = std::make_shared<Resize>(std::vector<int>{292});
  std::shared_ptr<TensorTransform> sr_centercrop = std::make_shared<CenterCrop>(std::vector<int>{256});
  std::shared_ptr<TensorTransform> sr_normalize =
    std::make_shared<Normalize>(std::vector<float>{123.68, 116.78, 103.94}, std::vector<float>{1.0, 1.0, 1.0});

  std::vector<std::shared_ptr<TensorTransform>> trans_list;

  if (FLAGS_network == "se-resnet50") {
    trans_list = {decode, sr_resize, sr_centercrop, sr_normalize, hwc2chw};
  } else if (FLAGS_network == "resnet101") {
    trans_list = {decode, resize, centercrop, normalizeResnet101, hwc2chw};
  } else {
    trans_list = {decode, resize, centercrop, normalize, hwc2chw};
  }
  mindspore::dataset::Execute SingleOp(trans_list);

  for (size_t i = 0; i < size; ++i) {
    struct timeval start = {0};
    struct timeval end = {0};
    double startTimeMs;
    double endTimeMs;
    std::vector<MSTensor> outputs;
    std::cout << "Start predict input files:" << all_files[i] << std::endl;

    MSTensor image = ReadFileToTensor(all_files[i]);
    if (FLAGS_dataset == "imagenet") {
      SingleOp(image, &image);
    }

    if (modelInputs[0].DataSize() < image.DataSize()) {
      std::cout << "ERROR: input data sizes do not match" << std::endl;
      return 1;
    }
    std::memcpy(modelInputs[0].MutableData(), image.Data().get(), image.DataSize());
    gettimeofday(&start, nullptr);
    ret = model.Predict(modelInputs, &outputs);
    gettimeofday(&end, nullptr);
    if (ret != kSuccess) {
      std::cout << "Predict " << all_files[i] << " failed." << std::endl;
      return 1;
    }
    startTimeMs = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
    endTimeMs = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
    costTime_map.insert(std::pair<double, double>(startTimeMs, endTimeMs));
    WriteResult(all_files[i], outputs);
  }
  double average = 0.0;
  int inferCount = 0;

  for (auto iter = costTime_map.begin(); iter != costTime_map.end(); iter++) {
    average += iter->second - iter->first;
    inferCount++;
  }
  average = average / inferCount;
  std::stringstream timeCost;
  timeCost << "NN inference cost average time: " << average << " ms of infer_count " << inferCount << std::endl;
  std::cout << "NN inference cost average time: " << average << "ms of infer_count " << inferCount << std::endl;
  std::string fileName = "./time_Result" + std::string("/test_perform_static.txt");
  std::ofstream fileStream(fileName.c_str(), std::ios::trunc);
  fileStream << timeCost.str();
  fileStream.close();
  costTime_map.clear();
  return 0;
}
