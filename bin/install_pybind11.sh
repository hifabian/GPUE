#!/bin/bash
PB11_TAG="2.5.0"
THIS_PATH=$(dirname "$(readlink -f \"$0\")")
PB11_INSTALL_PATH=${THIS_PATH}/../third_party

if [ -d "$PB11_INSTALL_PATH/pybind11" ]; then
    echo "pybind11 found at ${PB11_INSTALL_PATH}"
else
    echo "Cloning pybind11 v=${PB11_TAG} to ${PB11_INSTALL_PATH}"
    git clone https://github.com/pybind/pybind11 ${PB11_INSTALL_PATH}/pybind11
fi

