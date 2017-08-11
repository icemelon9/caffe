// ----------------------------------------------------------------------
// Written by Haichen Shen
// ----------------------------------------------------------------------

#ifndef CAFFE_UTIL_NMS_HPP_
#define CAFFE_UTIL_NMS_HPP_

#include <cmath>

#include "caffe/common.hpp"

namespace caffe {

template <typename Dtype>
void nms_cpu(const Dtype* boxes, const int* order, int num_boxes,
             Dtype nms_overlap_thresh, int* keep_out, int* num_out);

/*void nms_gpu(const float* boxes, int nboxes, float nms_overlap_thresh,
  int* keep_out, int* num_out);*/

} // namespace caffe

#endif // CAFFE_UTIL_NMS_HPP_
