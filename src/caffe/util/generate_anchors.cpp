// ----------------------------------------------------------------------
// Written by Haichen Shen
// ----------------------------------------------------------------------

#include <cmath>

#include "caffe/util/generate_anchors.hpp"

namespace caffe {

namespace {

inline void whctrs(const vector<float>& anchor, float* w, float* h,
                   float* x_ctr, float* y_ctr) {
  *w = anchor[2] - anchor[0] + 1;
  *h = anchor[3] - anchor[1] + 1;
  *x_ctr = anchor[0] + 0.5 * (*w - 1);
  *y_ctr = anchor[1] + 0.5 * (*h - 1);
}

void ratio_enum(const vector<float>& anchor, const vector<float>& ratios,
                vector<vector<float> >* boxes) {
  float w, h, x_ctr, y_ctr;
  whctrs(anchor, &w, &h, &x_ctr, &y_ctr);
  float size = w * h;
  for (int i = 0; i < ratios.size(); ++i) {
    float size_ratio = size / ratios[i];
    float new_w = round(sqrt(size_ratio));
    float new_h = round(new_w * ratios[i]);
    boxes->push_back({x_ctr, y_ctr, new_w, new_h});
  }
}

void scale_enum(const vector<float>& box, const vector<float>& scales,
                vector<vector<float> >* boxes) {
  for (int i = 0; i < scales.size(); ++i) {
    boxes->push_back({box[0], box[1], box[2] * scales[i], box[3] * scales[i]});
  }
}

} // namespace

void GenerateAnchors(uint32_t base_size, const vector<float>& ratios,
                     const vector<float>& scales,
                     vector<vector<float> >* anchor_boxes) {
  vector<float> base_anchor = {0, 0, float(base_size - 1.),
                               float(base_size - 1.)};
  vector<vector<float> > ratio_boxes;
  ratio_enum(base_anchor, ratios, &ratio_boxes);
  for (int i = 0; i < ratio_boxes.size(); ++i) {
    scale_enum(ratio_boxes[i], scales, anchor_boxes);
  }
}

} // namespace caffe
