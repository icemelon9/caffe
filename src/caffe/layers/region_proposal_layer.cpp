// ----------------------------------------------------------------------
// Written by Haichen Shen
// ----------------------------------------------------------------------

#include <math.h>
#include <cmath>

#include "caffe/layers/region_proposal_layer.hpp"
#include "caffe/util/generate_anchors.hpp"
#include "caffe/util/nms.hpp"

namespace caffe {

template <typename Dtype>
void RegionProposalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  RegionProposalParameter proposal_param =
      this->layer_param_.region_proposal_param();
  CHECK_GT(proposal_param.feature_stride(), 0)
      << "feature_stride must > 0";
  feature_stride_ = proposal_param.feature_stride();
  pre_nms_top_n_ = proposal_param.pre_nms_top_n();
  post_nms_top_n_ = proposal_param.post_nms_top_n();
  nms_threshold_ = proposal_param.nms_threshold();
  min_size_ = proposal_param.min_size();
  global_context_ = proposal_param.global_context();
  // Compute the anchors
  vector<float> scales = {8, 16, 32};
  if (proposal_param.scale_size() > 0) {
    scales.resize(proposal_param.scale_size());
    for (int i = 0; i < proposal_param.scale_size(); ++i) {
      scales[i] = proposal_param.scale(i);
    }
  }
  vector<float> ratios = {0.5, 1, 2};
  if (proposal_param.ratio_size() > 0) {
    ratios.resize(proposal_param.ratio_size());
    for (int i = 0; i < proposal_param.ratio_size(); ++i) {
      ratios[i] = proposal_param.ratio(i);
    }
  }
  GenerateAnchors(feature_stride_, ratios, scales, &anchors_);
  num_anchors_ = anchors_.size();
}

template <typename Dtype>
void RegionProposalLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  batch_ = bottom[0]->shape(0);
  height_ = bottom[0]->shape(2);
  width_ = bottom[0]->shape(3);
  max_num_boxes_ = height_ * width_ * num_anchors_;
  // Reshape workspace
  proposals_.Reshape({max_num_boxes_ * 4});
  order_.Reshape({max_num_boxes_});
  if (pre_nms_top_n_ > 0) {
    nms_out_.Reshape({std::min(max_num_boxes_, pre_nms_top_n_)});
  } else {
    nms_out_.Reshape({max_num_boxes_});
  }
  // Reshape output
  int num_rois = nms_out_.shape(0); // num rois per input
  if (post_nms_top_n_ > 0) {
    num_rois = std::min(num_rois, post_nms_top_n_);
  }
  int num_rois_global = num_rois;
  if (global_context_) {
    ++num_rois_global;
  }
  top[0]->Reshape({batch_ * num_rois_global, 5});
  top[1]->Reshape({batch_});
  if (top.size() > 2) {
    top[2]->Reshape({batch_ * num_rois});
  }
}

template <typename Dtype>
void RegionProposalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  int in_size = height_ * width_;
  // scores shape is N * (2 * A) * H * W, where the first set of num_anchors_
  // channels are bg probs, and the second set are fg probs
  const Dtype* scores_data = bottom[0]->cpu_data();
  // bbox_deltas shape is N * (A * 4) * H * W
  const Dtype* bbox_deltas_data = bottom[1]->cpu_data();
  const Dtype* im_info = bottom[2]->cpu_data();
  Dtype im_height = im_info[0];
  Dtype im_width = im_info[1];
  Dtype im_scale = im_info[2];
  Dtype min_size = min_size_ * im_scale;
  // temporary buffer
  Dtype* proposals = proposals_.mutable_cpu_data();
  int* order = order_.mutable_cpu_data();
  int* nms_out = nms_out_.mutable_cpu_data();
  // output
  int total_rois = 0;
  Dtype* out_rois = top[0]->mutable_cpu_data();
  Dtype* out_num_proposals = top[1]->mutable_cpu_data();
  Dtype* out_scores = nullptr;
  if (top.size() > 2) {
    out_scores = top[2]->mutable_cpu_data();
  }

  if (global_context_) {
    for (int n = 0; n < batch_; ++n) {
      out_rois[0] = n;
      out_rois[1] = 0.;
      out_rois[2] = 0.;
      out_rois[3] = im_info[1] - 1;
      out_rois[4] = im_info[0] - 1;
      out_rois += 5;
    }
    total_rois += batch_;
  }
  
  for (int n = 0; n < batch_; ++n) {
    const Dtype* fg_scores = scores_data + bottom[0]->offset(n, num_anchors_);
    const Dtype* bbox_deltas = bbox_deltas_data + bottom[1]->offset(n);
    int num_props = 0;
    for (int y = 0; y < height_; ++y) {
      for (int x = 0; x < width_; ++x) {
        for (int a = 0; a < num_anchors_; ++a) {
          // 1. Generate proposals from bbox deltas and shifted anchors
          int delta_offset = a * 4 * in_size + y * width_ + x;
          Dtype dx = bbox_deltas[delta_offset];
          Dtype dy = bbox_deltas[in_size + delta_offset];
          Dtype dw = bbox_deltas[2 * in_size + delta_offset];
          Dtype dh = bbox_deltas[3 * in_size + delta_offset];
          Dtype x_ctr = anchors_[a][0] + x * feature_stride_ +
                        dx * anchors_[a][2];
          Dtype y_ctr = anchors_[a][1] + y * feature_stride_ +
                        dy * anchors_[a][3];
          Dtype w = exp(dw) * anchors_[a][2];
          Dtype h = exp(dh) * anchors_[a][3];
          // 2. clip predicted boxes to image
          Dtype x1 = std::max(std::min(
              Dtype(x_ctr - 0.5 * w), Dtype(im_width - 1.)), (Dtype)0.);
          Dtype y1 = std::max(std::min(
              Dtype(y_ctr - 0.5 * h), Dtype(im_height - 1)), (Dtype)0.);
          Dtype x2 = std::max(std::min(
              Dtype(x_ctr + 0.5 * w - 1), Dtype(im_width - 1)), (Dtype)0.);
          Dtype y2 = std::max(std::min(
              Dtype(y_ctr + 0.5 * h - 1), Dtype(im_height - 1)), (Dtype)0.);
          // use the same index in the fg_scores
          int prop_index = a * in_size + y * width_ + x;
          proposals[prop_index * 4] = x1;
          proposals[prop_index * 4 + 1] = y1;
          proposals[prop_index * 4 + 2] = x2;
          proposals[prop_index * 4 + 3] = y2;
          // 3. remove predicted boxes with either height or width < threshold
          // (NOTE: convert min_size to input image scale stored in im_info[2])
          if (w >= min_size && h >= min_size) {
            order[num_props++] = prop_index;
          }
        }
      }
    }

    // 4. sort all (proposal, score) pairs by score from highest to lowest
    // 5. take top pre_nms_topN (e.g. 6000)
    std::sort(order, order + num_props,
              [&](int a, int b) { return fg_scores[b] < fg_scores[a]; });
    if (pre_nms_top_n_ > 0 && num_props > pre_nms_top_n_) {
      num_props = pre_nms_top_n_;
    }

    // 6. apply nms (e.g. threshold = 0.7)
    // 7. take after_nms_topN (e.g. 300)
    int num_out;
    nms_cpu(proposals, order, num_props, nms_threshold_, nms_out, &num_out);
    if (post_nms_top_n_ > 0) {
      num_out = std::min(num_out, post_nms_top_n_);
    }
    // 8. return the top proposals (-> RoIs top)
    for (int i = 0; i < num_out; ++i) {
      int idx = nms_out[i];
      out_rois[0] = n; // batch index
      out_rois[1] = proposals[idx * 4];
      out_rois[2] = proposals[idx * 4 + 1];
      out_rois[3] = proposals[idx * 4 + 2];
      out_rois[4] = proposals[idx * 4 + 3];
      out_rois += 5;
      if (out_scores != nullptr) {
        *(out_scores++) = fg_scores[idx];
      }
    }
    out_num_proposals[n] = num_out;
    total_rois += num_out;
  }
  // reshape the top based on actual number of ROIs generated
  top[0]->Reshape({total_rois, 5});
  if (top.size() > 2) {
    top[2]->Reshape({total_rois});
  }
}

template <typename Dtype>
void RegionProposalLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  LOG(FATAL) << "RegionProposalLayer doesn't support back propagation";
}

#ifdef CPU_ONLY
STUB_GPU(RegionProposalLayer);
#endif

INSTANTIATE_CLASS(RegionProposalLayer);
REGISTER_LAYER_CLASS(RegionProposal);

} // namespace caffe
