# Interactive Image2Video-Synthesis

[Andreas Blattmann](https://www.linkedin.com/in/andreas-blattmann-479038186/?originalSubdomain=de),
[Timo Milbich](https://timomilbich.github.io/),
[Michael Dorkenwald](https://mdork.github.io/),
[Björn Ommer](https://hci.iwr.uni-heidelberg.de/Staff/bommer),
[CVPR 2021](http://cvpr2021.thecvf.com/)<br/>

This repository contains the code to reproduce the results presented in our paper [Understanding Object Dynamics for Interactive Image-to-Video Synthesis](toso add link) and train new models to enable human users to interact with still images.  


![teaser](images/overview.png "Overview over our model.")

[**Arxiv**](todo add arxiv link once up) | [**Project page**](https://compvis.github.io/interactive-image2video-synthesis/) | [**BibTex**]()

**tl;dr** We introdice the novel problem of Interactive Image-to-Video Synthesis where we learn to understand the relations between the distinct body parts of articulated objects from unlabeled video data. Our proposed model allows for synthesis of videos showing natural object dynamics as responses to targeted, local interactions.and, thus, enables human users to interact with still images by poking pixels.

## Table of contents ##
1. [Requirements](#Requirements)
2. [Data preparation](#data_prep)
3. [Pretrained Models](#pretrained)
4. [Train your own II2V model](#training)

## Requirements <a name="Requirements"></a>
A suitable conda environment named ``ii2v`` can be created with

````shell script
conda env create -f ii2v.yml 
conda activate ii2v
````

##Data preparation <a name="data_prep"></a>

### Get Flownet2 for optical flow estimation ###

As preparing the data to evaluate our pretrained models or train new ones requires to estimate optical flow maps, first add [Flownet2](https://github.com/NVIDIA/flownet2-pytorch) as a git submodule and place it in the directory ``models/flownet2`` via

```shell script
git submodule add https://github.com/NVIDIA/flownet2-pytorch models/flownet2
``` 

Since Flownet2 requires cuda-10.0 and is therefore not compatible with our main conda environment, we provide a separate conda enviroment for optical flow estimation which can bet created via

```shell script
conda env create -f flownet2
```
You can activate the environment and specify the right cuda version by using 

```shell script
source activate_flownet2
``` 
from the root of this repository. IMPORTANT: You have to ensure that lines 3 and 4 in the script add your respective ``cuda-10.0`` installation direcories to the ``PATH`` and ``LD_LIBRARY_PATH`` environment variables.
Finally, you have to build the custom layers of flownet2 with

```shell script
cd models/flownet2
bash install.sh -ccbin <PATH TO_GCC7>
```
, where ``<PATH TO_GCC7>`` is the path to your ``gcc-7``-binary, which is usually ``/usr/bin/gcc-7`` on a linux server. Make sure that your ``flownet2`` environment is activated and that the env-variables contain the ``cuda-10.0`` installation when running the script.
   

### Poking Plants ###

Download Poking Plants dataset from [here](todo add link) and extract it to a ``<TARGETDIR>``, which then contains the raw video files. To extract the individual frames and estimate optical flow, run

````shell script
source activate_flownet2
python -m utils.prepare_dataset --config config/data_preparation/plants.yaml
````

### iPER ###

### Human3.6m ###

### TaiChi-HD ###

## Pretrained models <a name="pretrained"></a>

## Train your own II2V model <a name="training"></a>

