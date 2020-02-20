# Efficient-Monocular-Depth-Estimation-for-Edge-Devices-in-Internet-of-Things

## Contents
0. [Introduction](#introduction)
0. [Quick Guide](#quick-guide)
0. [Depth estimation models](#models)
0. [Results](#results)
0. [Citation](#citation)

## Introduction
The code and CNN models verify the results in our paper. The models for efficient depth estimation are available in the directory "results". The results from RGB images are the same as that in the paper when inputs are images on the NYU Depth V2<a href="http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz">testing dataset</a>. Additionally, our provided code can be used for inference on arbitrary images.

## Quick Guide
Our models can be run with Python 3.6 and PyTorch 1.0/1.2.
We use the preprocessed dataset <a href="http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz">NYU Depth V2</a> that are provided by Fangchang Ma and used by the paper<a href="https://github.com/fangchangma/sparse-to-dense.pytorch"> "Sparse-to-Dense: Depth Prediction from Sparse Depth Samples and a Single Image"</a>.
```bash
sudo apt-get install -y libhdf5-serial-dev hdf5-tools
pip3 install matplotlib h5py scikit-image imageio opencv-python
```
Additionally, please install <a href="https://docs.tvm.ai/install/index.html">TVM</a> on edge devices, if you need the results of the optimized MDE. Here, we adopted the TVM-0.5, LLVM-4.0, and CUDA-10.0. In detail, on TX2 CPU, TVM-0.5 and LLVM-4.0 were installed. On UP Board CPU, TVM-0.5 and LLVM-4.0 were installed. On TX2 GPU, TVM-0.5 and CUDA-10.0 were installed. On Nano GPU, TVM-0.5 and CUDA-10.0 were installed. 

## Depth estimation models
Our models can be downloaded in the directory: results/Dataset=nyudepth.nsample=0.lr=0.01.bs=1.optimizer=sgd. The MDE model is mobilenetv2blconv7dw_0.597.pth.tar. The pruned MDE model is mobilenetv2blconv7dw_0.579.pth.tar.

## Results
If you need the accuracy of the pruned MDE, please run the file main.py. The command is
```bash
python main.py -b 1 -s 0 --data /
/home/star/data/nyudepthv2 --epochs 30 --optimize sgd --activation relu --dataset nyudepth --lr 0.01 --evaluate  
```

If you need the accuracy of MDE, please modify the line (the file main.py) which is best_model_filename = os.path.join(output_directory, 'mobilenetv2blconv7dw_0.597.pth.tar').
The line should be changed to the following line: best_model_filename = os.path.join(output_directory, 'mobilenetv2blconv7dw_0.579.pth.tar').  
Then, run the following command and you can get the results of MDE.
```bash
python main.py -b 1 -s 0 --data /
/home/star/data/nyudepthv2 --epochs 30 --optimize sgd --activation relu --dataset nyudepth --lr 0.01 --evaluate  
```

If you need the runtime of MDE, pruned MDE, and other models without optimization, please run the following command.
```bash
python runtime.py
```

If you need our runtime of optimized models on edge devices, please run the following command.
```bash
python tune_run.py 
```
For example, on the device Jetson Nano, we obtained the runtime of the optimized models and three files: deploy_graph.json, deploy_lib.tar, and deploy_param.params. The image "UP Board_CPU_results.png" displays our runtime on UP Board CPU.  The image "Nano_GPU_results.jpg" displays our runtime on Nano GPU.
#### Citation
If you use our method or code in your work, please consider citing our paper.
The citation will be available after the paper is published.

