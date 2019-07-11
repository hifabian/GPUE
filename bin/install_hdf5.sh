HDF_VER_NO_PATCH="1.10"
HDF_VER="1.10.5"
ROOT_PATH="$PWD"
LOCAL_HDF_PATH="CMake-hdf5-$HDF_VER/hdf5-$HDF_VER/include"

if [ -d "./$LOCAL_HDF_PATH" ]; then
    echo "$PWD/$LOCAL_HDF_PATH"
    exit 0
fi

echo "Fetching HDF5 from remote"
wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-$HDF_VER_NO_PATCH/hdf5-$HDF_VER/src/CMake-hdf5-$HDF_VER.tar.gz
echo "Got archive"

echo "Extracting..."
tar -xvf "CMake-hdf5-$HDF_VER.tar.gz" && rm "CMake-hdf5-$HDF_VER.tar.gz"
echo "Finished Extraction"

cd "CMake-hdf5-$HDF_VER/hdf5-$HDF_VER"

echo "Configuring..."
./configure --prefix=$(pwd) --enable-cxx
echo "Finished Configuring"

echo "Building..."
make || cp bin/libszip-static.a bin/libszip.a # work around apparent bug in their makefile
make && make install
echo "Finished Building"

echo "$ROOT_PATH/$LOCAL_HDF_PATH"
exit 0
