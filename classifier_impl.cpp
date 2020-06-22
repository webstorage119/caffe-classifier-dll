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

#include "caffe_classifier.hpp"
#include "classifier_impl.hpp"

CaffeClassifier* CaffeClassifier::NewClassifier(const std::string& model_file, 
                                                const std::string& trained_file,
                                                const float mean,
                                                const float scale) {
  return new ClassifierImpl(model_file, trained_file, mean, scale);
}

ClassifierImpl::ClassifierImpl(const std::string& model_file, const std::string& trained_file, 
                               const float mean, const float scale): mean_(mean), scale_(scale) {
#ifdef CPU_ONLY
  caffe::Caffe::set_mode(caffe::Caffe::CPU);
#else
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
#endif
  net_.reset(new caffe::Net<float>(model_file, caffe::TEST));
  net_->CopyTrainedLayersFrom(trained_file);
  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";
  caffe::Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int>> pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], static_cast<int>(i)));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

std::vector<std::vector<float>> ClassifierImpl::Predict(const std::vector<cv::Mat>& imgs) {
  std::vector<std::vector<float>> result;
  if (imgs.empty()) {
    return result;
  }
  caffe::Blob<float>* input_layer = net_->input_blobs()[0];
#ifndef CPU_ONLY
  input_layer->Reshape(imgs.size(), num_channels_,
                       input_geometry_.height, input_geometry_.width);
  net_->Reshape();
#endif
  for (size_t i = 0; i < imgs.size(); ++i) {
    std::vector<cv::Mat> input_channels;
#ifdef CPU_ONLY
    input_layer->Reshape(1, num_channels_,
                         input_geometry_.height, input_geometry_.width);
    net_->Reshape();
    WrapInputLayer(0, &input_channels);
    Preprocess(imgs[i], &input_channels);
    net_->Forward();
    caffe::Blob<float>* out_layer = net_->output_blobs()[0];
    const float* begin = out_layer->cpu_data();
    const float* end = begin + out_layer->channels();
    result.emplace_back(std::vector<float>(begin, end));
#else
    WrapInputLayer(i, &input_channels);
    Preprocess(imgs[i], &input_channels);
#endif
  }
#ifndef CPU_ONLY
  net_->Forward();
  caffe::Blob<float>* out_layer = net_->output_blobs()[0];
  const float* pred = out_layer->cpu_data();
  for (size_t i = 0; i < imgs.size(); ++i) {
    const float* begin = pred + i * out_layer->channels();
    const float* end = begin + out_layer->channels();
    result.emplace_back(std::vector<float>(begin, end));
  }
#endif
  return result;
}

std::vector<Prediction> ClassifierImpl::Classify(const std::vector<cv::Mat>& imgs, int N) {
  std::vector<std::vector<float>> output = Predict(imgs);
  std::vector<Prediction> result;
  for (size_t i = 0; i < output.size(); ++i) {
    std::vector<int> maxN = Argmax(output[i], N);
    Prediction predictions;
    for (int i = 0; i < N; ++i) {
      int idx = maxN[i];
      predictions.push_back(std::make_pair(idx, output[i][idx]));
    }
    result.push_back(std::move(predictions));
  }
  return result;
}

void ClassifierImpl::WrapInputLayer(int n, std::vector<cv::Mat>* input_channels) {
  caffe::Blob<float>* input_layer = net_->input_blobs()[0];
  float* input_data = input_layer->mutable_cpu_data() +
    n * num_channels_ * input_geometry_.area();
  for (int i = 0; i < num_channels_; ++i) {
    cv::Mat channel(input_geometry_.height, input_geometry_.width, 
                    CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += input_geometry_.area();
  }
}

void ClassifierImpl::Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels) {
  cv::Mat sample;
  cv::resize(img, sample, input_geometry_);
  img.convertTo(sample, CV_32FC(num_channels_), scale_, - mean_ * scale_);
  cv::split(sample, *input_channels);
}