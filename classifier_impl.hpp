/*  Copyright (C) <2020>  <Yong WU>
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.
    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once
#include "caffe_classifier.hpp"
#include "caffe/caffe.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <memory>
#include <vector>
#include <string>


class ClassifierImpl : public CaffeClassifier {
public:
  ClassifierImpl(const std::string& model_file, const std::string& trained_file,
                 const float mean, const float scale);
  std::vector<Prediction> Classify(const std::vector<cv::Mat>& imgs, int N) override;

private:
  std::vector<std::vector<float>> Predict(const std::vector<cv::Mat>& imgs);
  void WrapInputLayer(int n, std::vector<cv::Mat>* input_channels);
  void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);

  std::shared_ptr<caffe::Net<float>> net_;
  int num_channels_;
  cv::Size input_geometry_;
  float mean_;
  float scale_;
};