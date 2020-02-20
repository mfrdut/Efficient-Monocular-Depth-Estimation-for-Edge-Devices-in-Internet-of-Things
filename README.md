# Efficient-Monocular-Depth-Estimation-for-Edge-Devices-in-Internet-of-Things

## Contents
0. [Introduction](#introduction)
0. [Quick Guide](#quick-guide)
0. [Depth estimation models](#models)
0. [Results](#results)
0. [Citation](#citation)

## Introduction
This part includes pretrained models, which are stored in <a href="https://drive.google.com/file/d/1heAXjHVK0yQ4oKyR0qIyY4sRfSA_CapN/view?usp=sharing">Google Drive</a>.

The code and models verify the results in this paper "Efficient Monocular Depth Estimation for Edge Devices in Internet of Things".
The CNN models for efficient depth estimation are available in this directory of results. The results from RGB images are the same as that in the paper when inputs are the testing dataset in <a href="http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz">datasets</a>. Additionally, our provided code can be used for inference on arbitrary images.

## Quick Guide
This models can be run with Python 3.6 and PyTorch 1.0 or PyTorch 1.2.
Download the preprocessed dataset <a href="http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz">NYU Depth V2</a> and <a href="http://datasets.lids.mit.edu/sparse-to-dense/data/kitti.tar.gz">KITTI Odometry dataset</a> that are provided by Fangchang Ma and used by the paper<a href="https://github.com/fangchangma/sparse-to-dense.pytorch"> "Sparse-to-Dense: Depth Prediction from Sparse Depth Samples and a Single Image"</a>.
```bash
sudo apt-get install -y libhdf5-serial-dev hdf5-tools
pip3 install matplotlib h5py scikit-image imageio opencv-python
```
## Depth estimation models
Our models can be downloaded in the above directory: results\/Dataset=nyudepth.nsample=0.lr=0.01.bs=1.optimizer=sgd. These models are used to acquire the results reported in our paper on the benchmark datasets NYU-Depth-v2 for efficient depth estimation. The MDE model is mobilenetv2blconv7dw_0.597.pth.tar. The pruned MDE model is mobilenetv2blconv7dw_0.579.pth.tar.

## Results
Run the file main.py to obtain the results of the pruned MDE in this paper "Efficient Monocular Depth Estimation for Edge Devices in Internet of Things". The command is:
```bash
python main.py -b 1 -s 0 --data /
/home/star/data/nyudepthv2 --epochs 30 --optimize sgd --activation relu --dataset nyudepth --lr 0.01 --evaluate  
```
When you need the results of MDE, please modify the line (the file main.py) which is:
best_model_filename = os.path.join(output_directory, 'mobilenetv2blconv7dw_0.597.pth.tar').
The line should be changed to the following line:
best_model_filename = os.path.join(output_directory, 'mobilenetv2blconv7dw_0.579.pth.tar').  
Run the following command and you can get the results of MDE.
```bash
python main.py -b 1 -s 0 --data /
/home/star/data/nyudepthv2 --epochs 30 --optimize sgd --activation relu --dataset nyudepth --lr 0.01 --evaluate  
```
#### Citation
If you use our method or code in your work, please consider citing our paper.
The citation will be available after the paper is published.

