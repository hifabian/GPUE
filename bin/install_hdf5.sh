HDF_VER_NO_PATCH="1.10"
HDF_VER="1.10.5"
ROOT_PATH="$PWD"
LOCAL_HDF_PATH="CMake-hdf5-$HDF_VER/HDF5-$HDF_VER-Linux/HDF_Group/HDF5/$HDF_VER/include"

if [[ ! -z "$HDF5_DIR" ]]; then
    echo "$HDF5_DIR"
    exit 0
fi

if [ -d "./$LOCAL_HDF_PATH" ]; then
    echo "$PWD/$LOCAL_HDF_PATH"
    exit 0
fi

echo "Fetching HDF5 from remote"
wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-$HDF_VER_NO_PATCH/hdf5-$HDF_VER/src/CMake-hdf5-$HDF_VER.tar.gz
echo "Got HDF5 archive"

echo "Extracting HDF5..."
tar -xvf "CMake-hdf5-$HDF_VER.tar.gz"
echo "Extracted HDF5"

echo "Running build script"
cd "CMake-hdf5-$HDF_VER/"
bash "./build-unix.sh"
echo "Build step complete"

echo "Extracting built source"
tar -xvf "HDF5-$HDF_VER-Linux.tar.gz"
echo "Extracted source"
echo "$ROOT_PATH/$LOCAL_HDF_PATH"
exit 0
