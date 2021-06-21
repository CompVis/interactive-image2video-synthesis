# Interactive Image2Video-Synthesis

[Andreas Blattmann](https://www.linkedin.com/in/andreas-blattmann-479038186/?originalSubdomain=de),
[Timo Milbich](https://timomilbich.github.io/),
[Michael Dorkenwald](https://mdork.github.io/),
[Björn Ommer](https://hci.iwr.uni-heidelberg.de/Staff/bommer),
[CVPR 2021](http://cvpr2021.thecvf.com/)<br/>

=======
This repository contains the code to reproduce the results presented in our paper [Understanding Object Dynamics for Interactive Image-to-Video Synthesis](toso add link) and train new models to enable human users to interact with still images.  


![teaser](images/overview.png "Overview over our model.")

[**Arxiv**](todo add arxiv link once up) | [**Project page**](https://compvis.github.io/interactive-image2video-synthesis/) | [**BibTex**]()

**TL;DR** We introduce the novel problem of Interactive Image-to-Video Synthesis where we learn to understand the relations between the distinct body parts of articulated objects from unlabeled video data. Our proposed model allows for synthesis of videos showing natural object dynamics as responses to targeted, local interactions.and, thus, enables human users to interact with still images by poking pixels.

## Table of contents ##
1. [Requirements](#Requirements)
2. [Data preparation](#data_prep)
3. [Pretrained Models](#pretrained)
4. [Train your own II2V model](#training)
5. [BibTeX](#bibtex)

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

Download Poking Plants dataset from [here](https://heibox.uni-heidelberg.de/d/71de55de923646509bc4/) and extract it to a ``<TARGETDIR>``, which then contains the raw video files. 
To extract the multi-zip file, use 

```shell script
zip -s 0 poking_plants.zip --out poking_plants_unsplit.zip
unzip poking_plants_unsplit.zip
```

To extract the individual frames and estimate optical flow set the value of the field 
``raw_dir`` in ``config/data_preparation/plants.yaml`` to be ``<TARGETDIR>``, define the target location for the extracted frames (, where all frames of each video will be within a unique directory) via the field ``processed_dir`` and run

````shell script
source activate_flownet2
python -m utils.prepare_dataset --config config/data_preparation/plants.yaml
````
By defining the number of parallel runs of flownet2, which will be distributed among the gpus with the ids specified in ``target_gpus``, with the ``num_workers``-argument, you can significantly speed up the optical flow estimation.  
### iPER ###

Download the zipped videos in ```iPER_1024_video_release.zip``` from [this website](https://onedrive.live.com/?authkey=%21AJL%5FNAQMkdXGPlA&id=3705E349C336415F%2188052&cid=3705E349C336415F) 
website (note that you have to create a microsoft account to get access) and extract the archive to a ```<TARGETDIR>``` similar to the above example. There, you'll also find the ``train.txt`` and ``val.txt``. Download these files and save them in the ``<TARGETDIR>`` 
Again, set the undefined value of the field ``raw_dir`` in ``config/data_preparation/iper.yaml`` to be ``<TARGETDIR>``, define the target location for the extracted frames and the optical flow via ``processed_dir`` and run 
```shell script
python -m utils.prepare_dataset --config config/data_preparation/iper.yaml
``` 
with the ````flownet2```` environment activated. 

### Human3.6m ###

Firstly, you will need to create an account at [the homepage of the Human3.6m dataset](http://vision.imar.ro/human3.6m/) to gain access to the dataset. After your account is created and approved (takes a couple of hours), log in and inspect your cookies to find your `PHPSESSID`. 
Fill in that `PHPSESSID` in `data/config.ini` and also specify the `TARGETDIR` there, where the extracted videos will be later stored. After setting the field `processed_dir` in `config/data_preparation/human36m.yaml`, you can download and extract the videos via
```shell script
python -m data.human36m_preprocess
```
with the ````flownet2```` environment activated. 
Frame extraction and optical flow estimation are then done as usual with
```shell script
python -m data.prepare_dataset --config config/data_preparation/human36m.yaml
```

### TaiChi-HD ###

To download and extract the videos, follow the steps listed at the [download page](https://github.com/AliaksandrSiarohin/first-order-model/tree/master/data/taichi-loading) for this dataset and set the `out_folder` argument of the script `load_videos.py` to be our `<TARGETDIR>` from the above examples. Again set the fields `raw_dir` and `processed_dir` in `config/data_preparation/taichi.yaml` similar to the above examples and run
```shell script
python -m data.prepare_dataset --config config/data_preparation/taichi.yaml
```
with the `flownet2` environment activated to extract the individual frames and estimate the optical flow maps.
## Pretrained models <a name="pretrained"></a>

### Get the checkpoints ###

 Here's a list of all available pretrained models. Note that the list will be updated soon, as we then also provide the pretrained models for the additional examples in [the supplementary](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Blattmann_Understanding_Object_Dynamics_CVPR_2021_supplemental.zip)

| Dataset  | Video resolution | Link |  FVD 
|----------|----------|----------|--------- |
| Poking Plants | 128 x 128 | [plants_128x128](https://heibox.uni-heidelberg.de/d/25d9afb4743446709f73/) | 174.18 |
| Poking Plants | 64 x 64 | [plants_64x64](https://heibox.uni-heidelberg.de/d/0ae26899aed6443ebdec/) | 89.76 |
| iPER | 128 x 128 | [iper_128x128](https://heibox.uni-heidelberg.de/d/0695ee70557c4f90bcbe/) | 220.34 |
| iPER | 64 x 64 | [iper_64x64](https://heibox.uni-heidelberg.de/d/8486eafdfea2405d9ead/) | 144.92 |
| Human3.6m | 128 x 128 | [h36m_128x128](https://heibox.uni-heidelberg.de/d/1956b6e6afbb4bb681d2/) | 129.62 |
| Human3.6m | 64 x 64 | [h36m_64x64](https://heibox.uni-heidelberg.de/d/db59ab4cd2624dce99ed/)| 119.89 |
| TaiChi-HD | 128 x 128 | [taichi_128x128](https://heibox.uni-heidelberg.de/d/98d376baafe64a828093/) | 167.94 |
| TaiChi-HD | 64 x 64 | [taichi_64x64](https://heibox.uni-heidelberg.de/d/2b7873d9620642d28c21/) | 182.28 |

Download the data to a `<MODELDIR>` by selecting all items visible under the respective link and clicking on the green 'ZIP Selected Items'. **IMPORTANT:** To ensure smooth and automatic evaluation, choose the name for the resulting zip-file to be the name of the respective link in the above table.

### Evaluate pretrained models ###

All provided pretrained models can be evaluated with the command
```shell script
conda activate ii2v
python -m utils.eval_pretrained --base_dir <MODELDIR> --mode <[metrics,fvd]> --gpu <GPUID>
``` 
, where `--mode fvd` will extract samples for calculating the FVD score (for details on its calculation see below) and save them in `<MODELDIR>/<NAME OF LINK IN TABLE>/generated/samples_fvd` and `--mode metrics` will evaluate the model wrt. the remaining metrics which we reported in the paper.

### FVD evaluation ###
As the FVD implementation requires `tensorflow<=1.15`, we again created a separate conda environment to evaluate the models wrt. the this score, which can be initialized 
and activated by using 
```shell script
conda env create -f environement_fvd.yml
conda activate fvd
``` 
You can calculate the FVD-score of a model with
 ```shell script
python -m utils.metric_fvd --gpu <GPUID> --source <MODELDIR>/<NAME OF LINK IN TABLE>/generated/samples_fvd
```
Note that the samples have to be written to `<MODELDIR>/<NAME OF LINK IN TABLE>/generated/samples_fvd` when running the script. 

## Train your own II2V model <a name="training"></a>

To train your own model on one of the provided datasets, you'll have to adapt the fields
* `base_dir` : The base directory where all logs, config-files, checkpoints and results will be stored (we recommend not to change this once you've defined it) 
* `dataset` : The considered dataset, shall be in `['PlantDataset, IperDataset, Human36mDataset, TaichiDataset]`
* `datapath`: `<TARGETDIR>` from above for the respective dataset

in the config file `config/fixed_length_model.yaml`.  

After that, you can start training by running
```shell script
python main.py --config config/fixed_length_model.yaml --project_name <UNIQUE_PROJECT_NAME> --gpu <GPUID> --mode <[train, test]>.
```

To evaluate the model after training, run 

```shell script
python -m utils.eval_models --base_dir <base_dir field from the respective config> --mode <[metrics,fvd]> --gpu <GPUID>
```



## BibTeX <a name="bibtex"></a>

```
@InProceedings{Blattmann_2021_CVPR,
    author    = {Blattmann, Andreas and Milbich, Timo and Dorkenwald, Michael and Ommer, Bjorn},
    title     = {Understanding Object Dynamics for Interactive Image-to-Video Synthesis},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {5171-5181}
}
```
