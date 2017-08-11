#include "caffe/layers/region_proposal_layer.hpp"

namespace caffe {

template <typename Dtype>
void RegionProposalLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

template <typename Dtype>
void RegionProposalLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  LOG(FATAL) << "RegionProposalLayer doesn't support back propagation";
}

INSTANTIATE_LAYER_GPU_FUNCS(RegionProposalLayer);

} // namespace caffe
