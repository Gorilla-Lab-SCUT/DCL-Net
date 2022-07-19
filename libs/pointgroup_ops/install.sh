#!/bin/bash
python setup.py build_ext  # if any head file not found add the include dir of this env to the CPLUS_INCLUDE_PATH
                           # eg: export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/home/anaconda3/envs/env_A/include
python setup.py develop