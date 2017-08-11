// ----------------------------------------------------------------------
// Written by Haichen Shen
// ----------------------------------------------------------------------

#include "caffe/layers/global_roi_layer.hpp"

namespace caffe {

namespace {

template <typename Dtype>
__global__ void global_roi_kernel(const Dtype* im_info, Dtype* out) {
  int idx = threadIdx.x;
  out[idx * 5] = (Dtype) idx;
  out[idx * 5 + 1] = 0.;
  out[idx * 5 + 2] = 0.;
  out[idx * 5 + 3] = im_info[1] - 1;
  out[idx * 5 + 4] = im_info[0] - 1;
}

template __global__ void global_roi_kernel(const float* im_info, float* out);
template __global__ void global_roi_kernel(const double* im_info, double* out);

} // namespace

template <typename Dtype>
void GlobalRoILayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  const Dtype* im_info = bottom[0]->gpu_data();
  Dtype* out = top[0]->mutable_gpu_data();
  global_roi_kernel<<<1, 1>>>(im_info, out);
}

template <typename Dtype>
void GlobalRoILayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  LOG(FATAL) << "GlobalRoILayer doesn't support back propagation";
}

INSTANTIATE_LAYER_GPU_FUNCS(GlobalRoILayer);

} // namespace caffe
