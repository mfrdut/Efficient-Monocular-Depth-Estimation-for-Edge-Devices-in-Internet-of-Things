# Efficient-Monocular-Depth-Estimation-for-Edge-Devices-in-Internet-of-Things

## Contents
0. [Introduction](#introduction)
0. [Quick Guide](#quick-guide)
0. [Depth estimation models](#models)
0. [Results](#results)
0. [Citation](#citation)

## Introduction
The code and CNN models verify the results in our paper. The models for efficient depth estimation are available in the directory "results". The results are the same as those in the paper when inputs are images on the NYU Depth V2<a href="http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz"> testing dataset</a>. Additionally, our provided code can be used for inference on arbitrary images.

## Quick Guide
Our models can be run with Python 3.6 and PyTorch 1.0/1.2.
We use the preprocessed dataset <a href="http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz">NYU Depth V2</a> that are provided by Fangchang Ma and used by the paper<a href="https://github.com/fangchangma/sparse-to-dense.pytorch"> "Sparse-to-Dense: Depth Prediction from Sparse Depth Samples and a Single Image"</a>.
```bash
sudo apt-get install -y libhdf5-serial-dev hdf5-tools
pip3 install matplotlib h5py scikit-image imageio opencv-python
pip install --upgrade git+https://github.com/mit-han-lab/torchprofile.git
```
Additionally, please install <a href="https://docs.tvm.ai/install/index.html">TVM</a> on edge devices, if you need the results of our optimization. Here, we adopted the TVM-0.5, LLVM-4.0, and CUDA-10.0. In detail, on TX2 CPU, TVM-0.5 and LLVM-4.0 were installed. On UP Board CPU, TVM-0.5 and LLVM-4.0 were installed. On TX2 GPU, TVM-0.5 and CUDA-10.0 were installed. On Nano GPU, TVM-0.5 and CUDA-10.0 were installed. 
To this end, firstly, build the Shared Library and clone TVM repo from its github:
```bash
sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
git clone https://github.com/apache/incubator-tvm tvm
cd tvm
git checkout v0.5
git submodule init
git submodule update
mkdir build
cp cmake/config.cmake build
```
Edit build/config.cmake to customize the compilation options
```bash
set(USE_CUDA OFF) -> set(USE_CUDA [path_to_cuda]) # e.g. /Users/txh/cuda-10.0/
set(USE_LLVM OFF) -> set(USE_LLVM [path_to_llvm-config]) # e.g. /Users/txh/llvm-4.0/bin/llvm-config
```
Build tvm and related libraries.
```bash
cd build
cmake ..
make -j8
```
Finally, update the PYTHONPATH environment variable and add the following lines in ~/.bashrc. 
```bash
export TVM_HOME=/path/to/tvm
export PYTHONPATH=$PYTHONPATH:~/tvm/python
```
## Depth estimation models
Our models can be downloaded in the directory: <a href="https://github.com/tutuxh/Efficient-Monocular-Depth-Estimation-for-Edge-Devices-in-Internet-of-Things/tree/master/results/Dataset%3Dnyudepth.nsample%3D0.lr%3D0.01.bs%3D1.optimizer%3Dsgd"> results/Dataset=nyudepth.nsample=0.lr=0.01.bs=1.optimizer=sgd</a>. The mobilenetv2blconv7dw_0.579.pth.tar is our original model. The mobilenetv2blconv7dw_0.597.pth.tar is our pruned model.

## Results
If you need the accuracy of our pruned model, please run the file main.py. The command is
```bash
python main.py -b 1 -s 0 --data /
/home/star/data/nyudepthv2 --epochs 30 --optimize sgd --activation relu --dataset nyudepth --lr 0.01 --evaluate  
```

If you need the accuracy of our original model, please modify the line (in the file main.py) which is "best_model_filename = os.path.join(output_directory, 'mobilenetv2blconv7dw_0.597.pth.tar')".
The line should be changed to the following line: best_model_filename = os.path.join(output_directory, 'mobilenetv2blconv7dw_0.579.pth.tar').  
Then, run the following command and you can get the results of our original model.
```bash
python main.py -b 1 -s 0 --data /
/home/star/data/nyudepthv2 --epochs 30 --optimize sgd --activation relu --dataset nyudepth --lr 0.01 --evaluate  
```

If you need the MACs of our pruned model, our original model, and other models without optimization, please use the commands like those:
```bash
import torch
from torchvision.models import resnet50

resnet50 = resnet50()
inputs = torch.randn(1, 3, 224, 224)
from torchprofile import profile_macs
macs = profile_macs(resnet50, inputs)
```
Here, please ensure that you have installed the latest version of <a href="https://github.com/mit-han-lab/torchprofile">torchprofile</a>.

If you need the runtime of our pruned model, our original model, and other models without optimization, please run the following command.
```bash
python runtime.py
```

If you need the runtime of our optimized models on edge devices, please run the following command.
```bash
python tune_run.py 
```
For example, on the device Jetson Nano, we obtained the runtime of the optimized models and three files: deploy_graph.json, deploy_lib.tar, and deploy_param.params. The image "Nano_GPU_results.jpg" displays our runtime on Nano GPU. The image "UP Board_CPU_results.png" displays our runtime on UP Board CPU. The results on different hardware architectures are listed in our papers.
#### Citation
If you use our methods or code in your work, please consider citing our paper.
The citation will be available after the paper is published.

