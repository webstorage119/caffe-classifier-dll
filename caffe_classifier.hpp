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

#ifdef CAFFE_CLASSIFIER_LINK_SHARED
#if defined(__GUNC__) && __GNUC__ >= 4
#define CAFFE_CLASSIFIER_API __attribute__ ((bisubility("default")))
#elif defined(__GUNC__)
#define CAFFE_CLASSIFIER_API
#elif defined(_MSC_VER)
#if defined (CAFFE_CLASSIFIER_API_EXPORTS)
#define CAFFE_CLASSIFIER_API __declspec(dllexport)
#else
#define CAFFE_CLASSIFIER_API __delcspec(dllimport)
#endif
#else
#define CAFFE_CLASSIFIER_API
#endif
#else
#define CAFFE_CLASSIFIER_API
#endif

#include "opencv2/core/core.hpp"
#include <string>
#include <vector>
#include <utility>


using Prediction = std::vector<std::pair<int, float>>; // class, confidence

class CAFFE_CLASSIFIER_API CaffeClassifier {
public:
  static CaffeClassifier* NewClassifier(const std::string& model_file, 
                                        const std::string& trained_file,
                                        const float mean = 0.0f,
                                        const float scale = 1.0f);
  CaffeClassifier() = default;
  CaffeClassifier(const CaffeClassifier&) = delete;
  CaffeClassifier& operator=(const CaffeClassifier&) = delete;
  virtual ~CaffeClassifier() = default;

  virtual std::vector<Prediction> Classify(const std::vector<cv::Mat>& imgs, int N) = 0;
};