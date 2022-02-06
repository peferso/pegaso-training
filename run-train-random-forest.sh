#!/bin/bash

source ~/.profile_PEGASO

cd ${PEGASO_TRAIN_DIR%/*}

run.sh train-random-forest.py

cd ${OLDPWD}
