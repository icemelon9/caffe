// ----------------------------------------------------------------------
// Written by Haichen Shen
// ----------------------------------------------------------------------

#include "caffe/layers/global_roi_layer.hpp"

namespace caffe {

template <typename Dtype>
void GlobalRoILayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void GlobalRoILayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape({1, 5});
}

template <typename Dtype>
void GlobalRoILayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  const Dtype* im_info = bottom[0]->cpu_data();
  Dtype* out = top[0]->mutable_cpu_data();
  out[0] = 0;
  out[1] = 0;
  out[2] = 0;
  out[3] = im_info[1] - 1;
  out[4] = im_info[0] - 1;
}

template <typename Dtype>
void GlobalRoILayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  LOG(FATAL) << "GlobalRoILayer doesn't support back propagation";
}

#ifdef CPU_ONLY
STUB_GPU(GlobalRoILayer);
#endif

INSTANTIATE_CLASS(GlobalRoILayer);
REGISTER_LAYER_CLASS(GlobalRoI);

} // namespace caffe
