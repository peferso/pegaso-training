#!/bin/bash

source ~/.profile_PEGASO

if (( $# != 1 )); then
    echo "Invalid number of parameters."
    echo "Please, invoke this script providing the target python script as in the following example:"
    echo "${0} process.py"
else
    echo "Correct invocation: $0 $1"
fi

PYTHON_SCRIPT_DIR=${PEGASO_TRAIN_DIR}
PYTHON_SCRIPT=${1}
SCRIPTNAME=${0%.*} ; SCRIPTNAME=${SCRIPTNAME##*/}
LOGFILE="${PEGASO_TRAIN_DIR%/*}/logs/${SCRIPTNAME}_${1%.*}_"`date "+%Y-%m-%d_%H-%M-%S"`.log
LOGFILE_DET="${PEGASO_TRAIN_DIR%/*}/logs/${SCRIPTNAME}_${1%.*}_"`date "+%Y-%m-%d_%H-%M-%S"`_pyoutput.log

echo 'Running '${0}'... - cd to script folder...'| tee -a ${LOGFILE}

cd ${PYTHON_SCRIPT_DIR} 

echo 'Running '${0}'... - cd to script folder. EXIT: '$? |tee -a ${LOGFILE}; 

echo 'Running '${0}'... - Activating python environment...'|tee -a ${LOGFILE}; 

source ../bin/activate

echo 'Running '${0}'... - Activating python environment. EXIT: '$?| tee -a ${LOGFILE}; 

echo 'Running '${0}'... - Running script '${PYTHON_SCRIPT}' in the background'| tee -a ${LOGFILE}; 

python3 ${PYTHON_SCRIPT} >> ${LOGFILE_DET} 2>&1 

PID_RUN=$(ps aux | grep -m1 "python3 /home/pietari/PycharmProjects/cars/pegaso-collect/src/process.py" | awk -F ' ' '{print $2}')

echo 'Running '${0}'... - Running script '${PYTHON_SCRIPT}' in the background: PID='${PID_RUN}| tee -a ${LOGFILE}; 

echo Execution of $0 finished: `date "+%Y-%m-%d %H:%M:%S"` | tee -a ${LOGFILE}

echo Python output written to ${LOGFILE_DET} | tee -a ${LOGFILE}

cd ${OLDPWD}

