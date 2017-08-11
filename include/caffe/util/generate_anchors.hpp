// ----------------------------------------------------------------------
// Written by Haichen Shen
// ----------------------------------------------------------------------

#ifndef CAFFE_UTIL_GENERATE_ANCHORS_HPP_
#define CAFFE_UTIL_GENERATE_ANCHORS_HPP_

#include "caffe/common.hpp"

namespace caffe {

void GenerateAnchors(uint32_t base_size, const vector<float>& ratios,
                     const vector<float>& scales,
                     vector<vector<float> >* anchor_boxes);

} // namespace caffe

#endif // CAFFE_UTIL_GENERATE_ANCHORS_HPP_
