cd pointops
python3 setup.py install
cd ../
cd downsample
c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) downsample.cpp -o ../downsampling$(python3-config --extension-suffix)