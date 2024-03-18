# Handwriting Recognition (MNIST) - AdvPT Project WS 2023

This project is a fully-connected neural network developed in C++ for the recognition of handwritten digits found in the MNIST dataset.

## Project Structure

The project is structured as follows:

- `build.sh`: This script contains all necessary code to prepare/build the executables for the project.
- `mnist.sh`: This script triggers the training and testing of the neural network implementation.
- `src/`: This directory contains the source code for the project, including `image_loader.cpp` and `label_loader.cpp`.

## Building the Project

To build the project, run the `build.sh` script. This will create a `build` directory (if it doesn't exist), navigate into it, run `cmake` and `make` to build the project, and then compile the `image_loader.cpp` and `label_loader.cpp` files.

```shellscript
./build.sh
```

## Running the Project

To run the project, execute the `mnist.sh` script. This will trigger the training and testing of the neural network implementation.

```shellscript
./mnist.sh
```

## Additional Notes

- The project uses C++20 standard.
- The project is developed and tested in a Linux environment.
- The project is developed using the CLion 2023.3.4 IDE.ry.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
