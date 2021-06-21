#!/bin/bash/

conda deactivate
# requires the flownet2 conda environment to be created (see README.md)
conda activate flownet2

PATH = ${PATH}:/usr/local/bin/cuda-10.0 LD_LIBRARY_PATH=${LD_LIBRARY_PATH} python -m data.prepare_dataset --config ${1}
