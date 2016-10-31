#include <iostream>
#include <string>
#include <vector>

#include "google/protobuf/text_format.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using namespace caffe;

int find_layer_id(const NetParameter& net, string layer_name) {
  for (int layer_id = 0; layer_id < net.layer_size(); ++layer_id) {
    const LayerParameter& layer_param = net.layer(layer_id);
    if (layer_param.name() == layer_name) {
      return layer_id;
    }
  }
  return -1;
}

int main(int argc, const char** argv) {
  if (argc < 5) {
    std::cout << argv[0] << "origin_prefix retarget_prefix layer output_prefix"
              << std::endl;
    return 1; 
  }
  string origin_prefix(argv[1]);
  string retarget_prefix(argv[2]);
  string target_layer(argv[3]);
  string output_prefix(argv[4]);

  {
    NetParameter origin_net, retarget_net;
    ReadProtoFromTextFile(origin_prefix + ".prototxt", &origin_net);
    ReadProtoFromTextFile(retarget_prefix + ".prototxt", &retarget_net);
    int target_layer_id = find_layer_id(origin_net, target_layer);
    int remove_cnt = origin_net.layer_size() - target_layer_id - 1;
    for (int i = 0; i < remove_cnt; ++i) {
      origin_net.mutable_layer()->RemoveLast();
    }
    // remove the input layer from retarget net
    retarget_net.mutable_layer()->DeleteSubrange(0, 1);
    // reset the first layer bottom name
    retarget_net.mutable_layer(0)->set_bottom(
        0, origin_net.layer(target_layer_id).top(0));
    // merge all layers from retarget net into origin net
    origin_net.mutable_layer()->MergeFrom(retarget_net.layer());

    WriteProtoToTextFile(origin_net, output_prefix + ".prototxt");
    //std::cout << "Output text model to " << output_prefix << ".prototxt"
    //          << std::endl;
  }

  {
    NetParameter origin_net, retarget_net;
    ReadProtoFromBinaryFile(origin_prefix + ".caffemodel", &origin_net);
    ReadProtoFromBinaryFile(retarget_prefix + ".caffemodel", &retarget_net);
    int target_layer_id = find_layer_id(origin_net, target_layer);
    int remove_cnt = origin_net.layer_size() - target_layer_id - 1;
    origin_net.mutable_layer()->DeleteSubrange(target_layer_id+1, remove_cnt);

    // remove the input layer from retarget net
    retarget_net.mutable_layer()->DeleteSubrange(0, 1);
    // reset the first layer bottom name
    retarget_net.mutable_layer(0)->set_bottom(
        0, origin_net.layer(target_layer_id).top(0));
    // merge all layers from retarget net into origin net
    origin_net.mutable_layer()->MergeFrom(retarget_net.layer());

    WriteProtoToBinaryFile(origin_net, output_prefix + ".caffemodel");
    //std::cout << "Output binary model to " << output_prefix << ".caffemodel"
    //          << std::endl;
  }
  return 0;
}
