mkdir -p build

cd build
cmake ..
make -j$(nproc)
cd ..

g++ src/image_loader.cpp -std=c++20 -o build/ImageLoader

g++ src/label_loader.cpp -std=c++20 -o build/LabelLoader