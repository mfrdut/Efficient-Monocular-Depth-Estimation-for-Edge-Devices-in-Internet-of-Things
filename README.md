# Efficient-Monocular-Depth-Estimation-for-Edge-Devices-in-Internet-of-Things

## Contents
0. [Introduction](#introduction)
0. [Quick Guide](#quick-guide)
0. [Depth estimation models](#models)
0. [Results](#results)
0. [Citation](#citation)

## Introduction
The code and CNN models verify the results in our paper. The models for efficient depth estimation are available in this directory of results. The results from RGB images are the same as that in the paper when inputs are the testing dataset in <a href="http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz">datasets</a>. Additionally, our provided code can be used for inference on arbitrary images.

## Quick Guide
Our models can be run with Python 3.6 and PyTorch 1.0/1.2.
Download the preprocessed dataset <a href="http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz">NYU Depth V2</a> and <a href="http://datasets.lids.mit.edu/sparse-to-dense/data/kitti.tar.gz">KITTI Odometry dataset</a> that are provided by Fangchang Ma and used by the paper<a href="https://github.com/fangchangma/sparse-to-dense.pytorch"> "Sparse-to-Dense: Depth Prediction from Sparse Depth Samples and a Single Image"</a>.
```bash
sudo apt-get install -y libhdf5-serial-dev hdf5-tools
pip3 install matplotlib h5py scikit-image imageio opencv-python
```
Additionally, please install <a href="https://docs.tvm.ai/install/index.html">TVM</a> on the edge devices, if you need the results of the optimized MDE. Here, we adopt the TVM-0.5, LLVM-4.0, and CUDA-10.0. In detail, on TX2 CPU, LLVM-4.0 was installed. On UP Board CPU, LLVM-4.0 was installed. On TX2 GPU, CUDA-10.0 was installed.  On Nano GPU, CUDA-10.0 was installed. 

## Depth estimation models
Our models can be downloaded in the directory: results/Dataset=nyudepth.nsample=0.lr=0.01.bs=1.optimizer=sgd. These models are used to acquire the results in our paper on the benchmark datasets NYU-Depth-v2. The MDE model is mobilenetv2blconv7dw_0.597.pth.tar. The pruned MDE model is mobilenetv2blconv7dw_0.579.pth.tar.

## Results
If you need the results of the pruned MDE, please run the file main.py. The command is:
```bash
python main.py -b 1 -s 0 --data /
/home/star/data/nyudepthv2 --epochs 30 --optimize sgd --activation relu --dataset nyudepth --lr 0.01 --evaluate  
```

If you need the results of MDE, please modify the line (the file main.py) which is: best_model_filename = os.path.join(output_directory, 'mobilenetv2blconv7dw_0.597.pth.tar').
The line should be changed to the following line: best_model_filename = os.path.join(output_directory, 'mobilenetv2blconv7dw_0.579.pth.tar').  
Then, run the following command and you can get the results of MDE.
```bash
python main.py -b 1 -s 0 --data /
/home/star/data/nyudepthv2 --epochs 30 --optimize sgd --activation relu --dataset nyudepth --lr 0.01 --evaluate  
```

If you need our results of compilation optimization on edge devices, please run the following command.
```bash
python tune_run.py 
```
For example, on the device Jetson Nano, we obtained the runtime of optimized MDE and three files which are deploy_graph.json, deploy_lib.tar, and deploy_param.params.
#### Citation
If you use our method or code in your work, please consider citing our paper.
The citation will be available after the paper is published.

