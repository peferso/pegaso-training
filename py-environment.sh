#!/bin/bash
#
# Requires 
# * python and pip version installed
# * virtualenv
#
# We will point to a specific python3.7 version, then we need to install it 
# before creating the environment.
# 
# Steps to install a python3.7 from binaries in Ubuntu:
#
# 1. sudo apt update
# 
# 2. sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget
#
# 3. cd /tmp
#
# 4.1. Find out the latest or specific version in https://www.python.org/downloads/
#
# 4.2. Check out which is the link to the zip file or compressed tar file
#
# 4.3 wget https://www.python.org/ftp/python/3.7.13/Python-3.7.13.tgz
#
# 5. tar -xf Python-3.7.13.tgz
#
# 6. cd Python-3.7.13
#
# 7.1. installation_dir=${HOME}/projects/.localpythons/3.7.13 ; mkdir -p ${installation_dir}
#
# 7.2. ./configure --prefix=${installation_dir} # This step may take some time
#
# 8.1. sudo make
# 8.2. sudo make install # To keep this version as an additional version to the existing ones in the system
#
# 9. ${installation_dir}/bin/python3.7 --version # check the installation
#
# Now we need to install the virtualenv package from source code since there is an issue when we use pip in environments created 
# in ubuntu 20.04 LTS from https://github.com/pypa/pipenv/issues/4804 referring to specific python binaries
#
# 10. mkdir -p $HOME/projects/.localpythons/src
#
# 11. cd $HOME/projects/.localpythons/src
#
# 12. wget https://files.pythonhosted.org/packages/5f/6c/d44c403a54ceb4ec5179d1a963c69887d30dc5b300529ce67c05b4f16212/virtualenv-20.14.1.tar.gz
#
# 13. tar -xf virtualenv-20.14.1.tar.gz
#
# 14. cd virtualenv-20.14.1
#
# 15. sudo $installation_dir/bin/python3 setup.py install

installation_dir=${HOME}/projects/.localpythons/3.7.13

python_version=${installation_dir}/bin/python3

environment_name=${PWD##*/}_pyenv

$installation_dir/bin/virtualenv -p ${python_version} ${environment_name}

source ${environment_name}/bin/activate

#
# Install the required packages (they were obtained using: "pip freeze --local > requirements.txt")
#

pip install -r requirements.txt

