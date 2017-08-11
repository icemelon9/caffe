// ----------------------------------------------------------------------
// Written by Haichen Shen
// ----------------------------------------------------------------------

#include <cstring>

#include "caffe/util/nms.hpp"

namespace caffe {

template <typename Dtype>
void nms_cpu(const Dtype* boxes, const int* order, int num_boxes,
             Dtype nms_overlap_thresh, int* keep_out, int* num_out) {
  bool* suppressed = new bool[num_boxes];
  memset(suppressed, 0, sizeof(bool) * num_boxes);
  Dtype* areas = new Dtype[num_boxes];
  for (int i = 0; i < num_boxes; ++i) {
    int idx = order[i];
    Dtype x1 = boxes[idx * 4];
    Dtype y1 = boxes[idx * 4 + 1];
    Dtype x2 = boxes[idx * 4 + 2];
    Dtype y2 = boxes[idx * 4 + 3];
    areas[i] = (x2 - x1 + 1) * (y2 - y1 + 1);
  }
  int n = 0;
  for (int i = 0; i < num_boxes; ++i) {
    if (suppressed[i]) {
      continue;
    }
    int ii = order[i];
    keep_out[n++] = ii;
    Dtype ix1 = boxes[ii * 4];
    Dtype iy1 = boxes[ii * 4 + 1];
    Dtype ix2 = boxes[ii * 4 + 2];
    Dtype iy2 = boxes[ii * 4 + 3];
    for (int j = i + 1; j < num_boxes; ++j) {
      if (suppressed[j]) {
        continue;
      }
      int jj = order[j];
      Dtype xx1 = std::max(ix1, boxes[jj * 4]);
      Dtype yy1 = std::max(iy1, boxes[jj * 4 + 1]);
      Dtype xx2 = std::min(ix2, boxes[jj * 4 + 2]);
      Dtype yy2 = std::min(iy2, boxes[jj * 4 + 3]);
      Dtype w = std::max((Dtype)0., Dtype(xx2 - xx1 + 1));
      Dtype h = std::max((Dtype)0., Dtype(yy2 - yy1 + 1));
      Dtype inter = w * h;
      Dtype overlap = inter / (areas[i] + areas[j] - inter);
      if (overlap >= nms_overlap_thresh) {
        suppressed[j] = true;
      }
    }
  }
  *num_out = n;
  delete[] suppressed;
  delete[] areas;
}

template
void nms_cpu<float>(const float* boxes, const int* order, int num_boxes,
                    float nms_overlap_thresh, int* keep_out, int* num_out);

template
void nms_cpu<double>(const double* boxes, const int* order, int num_boxes,
                     double nms_overlap_thresh, int* keep_out, int* num_out);

} // namespace caffe
