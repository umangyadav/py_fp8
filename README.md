# Python logic to enumerate FP8 dtypes values

This follows logic of ONNX's https://onnx.ai/onnx/technical/float8.html

Can be used to enumerate all values in float and hex for all 4 types of Floating points

# CPP Logic that compares FP8 casting logic from PyTorch with MIGraphX's 

To build main.cpp file, 
1. mkdir build
2. cd build
3. cmake ..

Note that PyTorch logic doesn't use saturation and will return NaNs for any value that is not representable in FP8 range. MIGraphX logic has "Clip" as template parameter that can be used to saturate. 
