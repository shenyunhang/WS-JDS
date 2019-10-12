# Cyclic Guidance for Weakly Supervised Joint Detection and Segmentation

By [Yunhang Shen](), [Rongrong Ji](http://mac.xmu.edu.cn/rrji-en.html), [Yan Wang](http://www.ee.columbia.edu/~yanwang/), [Yongjian Wu](), [Liujuan Cao]().

CVPR 2019 Paper

This project is based on [Detectron](https://github.com/facebookresearch/Detectron).


## Introduction



## License

WS-JDS is released under the [Apache 2.0 license](https://github.com/shenyunhang/WS-JDS/blob/ws-jds/LICENSE). See the [NOTICE](https://github.com/shenyunhang/WS-JDS/blob/ws-jds/NOTICE) file for additional details.


## Citing WS-JDS

If you find WS-JDS useful in your research, please consider citing:

```
@inproceedings{Shen_2019_CVPR,
    author = {Yunhang Shen and Rongrong Ji and Yan Wang and Yongjian Wu and Liujuan Cao},
    title = {{Cyclic Guidance for Weakly Supervised Joint Detection and Segmentation}},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2019},
}   
```


## Installation

**Requirements:**

- NVIDIA GPU, Linux, Python2
- Caffe2 in pytorch v1.0.1, various standard Python packages, and the COCO API; Instructions for installing these dependencies are found below

### Caffe2

Clone the pytorch repository:

```
# pytorch=/path/to/clone/pytorch
git clone https://github.com/pytorch/pytorch.git $pytorch
cd $pytorch
git checkout v1.0.1
git submodule update --init --recursive
```

Install Python dependencies:

```
pip install -r $pytorch/requirements.txt
```

Build caffe2:

```
cd $pytorch
sudo USE_OPENCV=On USE_LMDB=On BUILD_BINARY=On python2 setup.py install
```


### Other Dependencies

Install the [COCO API](https://github.com/cocodataset/cocoapi):

```
# COCOAPI=/path/to/clone/cocoapi
git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
cd $COCOAPI/PythonAPI
# Install into global site-packages
make install
# Alternatively, if you do not have permissions or prefer
# not to install the COCO API into global site-packages
python setup.py install --user
```

Note that instructions like `# COCOAPI=/path/to/install/cocoapi` indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (`COCOAPI` in this case) accordingly.

Install the [pycococreator](https://github.com/waspinator/pycococreator):

```
pip install git+git://github.com/waspinator/pycococreator.git@0.2.0
```


### WS-JDS

Clone the WS-JDS repository:

```
# WS-JDS=/path/to/clone/WS-JDS
git clone https://github.com/shenyunhang/WS-JDS.git $WS-JDS
cd $WS-JDS
```

Install Python dependencies:

```
pip install -r requirements.txt
```

Set up Python modules:

```
make
```

Build the custom operators library:

```
mkdir -p build && cd build
cmake .. -DCMAKE_CXX_FLAGS="-isystem $pytorch/third_party/eigen -isystem $/pytorch/third_party/cub"
make
```


### Dataset Preparation
Please follow [this](https://github.com/shenyunhang/WS-JDS/blob/ws-jds/detectron/datasets/data/README.md#creating-symlinks-for-pascal-voc) to creating symlinks for PASCAL VOC.

Download MCG proposal from [here](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/mcg/) to detectron/datasets/data, and transform it to pickle serialization format:

```
cd detectron/datasets/data
tar xvzf MCG-Pascal-Main_trainvaltest_2007-boxes.tgz
cd ../../../
python tools/convert_mcg.py voc_2007_train detectron/datasets/data/MCG-Pascal-Main_trainvaltest_2007-boxes detectron/datasets/data/mcg_voc_2007_train.pkl
python tools/convert_mcg.py voc_2007_val detectron/datasets/data/MCG-Pascal-Main_trainvaltest_2007-boxes detectron/datasets/data/mcg_voc_2007_val.pkl
python tools/convert_mcg.py voc_2007_test detectron/datasets/data/MCG-Pascal-Main_trainvaltest_2007-boxes detectron/datasets/data/mcg_voc_2007_test.pkl
```


### Model Preparation

Download VGG16 model (VGG_ILSVRC_16_layers.caffemodel and VGG_ILSVRC_16_layers_deploy.prototxtt) and transform it to pickle serialization format:
```
cd $WS-JDS
mkdir -p model/
cd model
wget http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel
wget https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/ded9363bd93ec0c770134f4e387d8aaaaa2407ce/VGG_ILSVRC_16_layers_deploy.prototxt
cd ../
./scripts/convert_vgg16.sh
```

Download DeepLabv2_VGG16 and transform it to pickle serialization format:
```
cd $WS-JDS
cd model
wget http://liangchiehchen.com/projects/released/deeplab_aspp_vgg16/prototxt_and_model.zip
unzip prototxt_and_model.zip
cd ..
python tools/pickle_caffe_blobs.py --prototxt model/train.prototxt --caffemodel model/init.caffemodel --output model/init.pkl 
python tools/combine_deeplab_and_original_vgg16.py model/VGG_ILSVRC_16_layers_v1.pkl model/init.pkl model/vgg16_init.pkl
```

Noted that this requires to instal caffe1 separately, as caffe1 specific proto is removed in pytorch v1.0.1. 
See [this](https://github.com/pytorch/pytorch/commit/40109b16d0df8248bc01ad08c7ab615310c52d67).

You can download vgg16_init.pkl from this [link](https://1drv.ms/u/s!AodeRhn8mpxoh01lSlZsiNJC-gNP?e=Lgsw5f).

You may also need to modify the below config files to point TRAINING.WEIGHTS to vgg16_init.pkl.


## Quick Start: Using WS-JDS
```
./scripts/train_wsl.sh --cfg configs/voc_2007/ws-jds_VGG16-C5D_1x.yaml OUTPUT_DIR experiments/ws-jds_vgg16_voc2007_`date +'%Y-%m-%d_%H-%M-%S'`
```

### Result
The final model and log can be downloaded from [here](https://1drv.ms/u/s!AodeRhn8mpxoh1bcp7TnmBu31Gow?e=uhuUFT).

Noted that the results reported in the paper are based on Caffe2 in pytorch v0.4.1, while this repository is based on v1.0.1.

And upgrade it to v1.2.0 may reduce the performance by ~2% mAP on PASCAL VOC, which is weird.
