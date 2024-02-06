CC=/opt/rocm/llvm/bin/clang
CXX=/opt/rocm/llvm/bin/clang++
cd ./lib/llama.cpp
cmake -H. -Bbuild -DLLAMA_HIPBLAS=ON -DAMDGPU_TARGETS=gfx1030 -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DBUILD_SHARED_LIBS=ON
cmake --build build -- -j 16
mkdir -p ../../build/lib
cp build/common/libcommon.a ../../build/lib/libcommon.a
cp build/libllama.so ../../build/libllama.so