#!/bin/bash
set -x
set -e

~/Documents/caffe/build/tools/upgrade_net_proto_binary \
	model/VGG_ILSVRC_16_layers.caffemodel \
	model/VGG_ILSVRC_16_layers_v1.caffemodel

~/Documents/caffe/build/tools/upgrade_net_proto_text \
	model/VGG_ILSVRC_16_layers_deploy.prototxt \
	model/VGG_ILSVRC_16_layers_deploy_v1.prototxt

python ./tools/pickle_caffe_blobs.py \
	--prototxt model/VGG_ILSVRC_16_layers_deploy_v1.prototxt \
	--caffemodel model/VGG_ILSVRC_16_layers_v1.caffemodel \
	--output model/VGG_ILSVRC_16_layers_v1.pkl
