#!/bin/bash
python setup.py bdist_wheel
cd dist
pip install spconv*
cd ../
