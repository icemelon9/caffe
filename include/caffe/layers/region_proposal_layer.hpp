// ----------------------------------------------------------------------
// Written by Haichen Shen
// ----------------------------------------------------------------------

#ifndef CAFFE_REGION_PROPOSAL_LAYER_HPP_
#define CAFFE_REGION_PROPOSAL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class RegionProposalLayer : public Layer<Dtype> {
 public:
  explicit RegionProposalLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Proposal"; }

  virtual inline int MinBottomBlobs() const { return 3; }
  virtual inline int MaxBottomBlobs() const { return 3; }
  virtual inline int MinTopBlobs() const { return 2; }
  virtual inline int MaxTopBlobs() const { return 3; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  // region proposal layer params
  int feature_stride_;
  int pre_nms_top_n_;
  int post_nms_top_n_;
  Dtype nms_threshold_;
  int min_size_;
  // input and output shapes
  int num_anchors_;
  int batch_;
  int height_;
  int width_;
  int max_num_boxes_;
  // anchors in (x_ctr, y_ctr, w, h) layout
  vector<vector<float> > anchors_;
  Blob<Dtype> proposals_;
  Blob<int> order_;
  Blob<int> nms_out_;
};

} // namesapce caffe

#endif // CAFFE_REGION_PROPOSAL_LAYER_HPP_
