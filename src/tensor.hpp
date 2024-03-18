
#pragma once

#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include <fstream>
#include <sstream>
#include <cassert>
#include <utility>

inline size_t flatIdx(const std::vector< size_t >& shape, const std::vector< size_t >& idx)
{
    assert(shape.size() == idx.size());

    auto rank = idx.size();

    if (rank == 0)
    {
        return 0;
    }
    else if (rank == 1)
    {
        return idx[0];
    }
    else if (rank == 2)
    {
        return idx[0] * shape[1] + idx[1];
    }
    else
    {
        size_t flatIdx = 0;
        for (size_t i = 0; i < rank; i++)
        {
            size_t dimProduct = 1;
            for (size_t ii = i + 1; ii < rank; ii++)
            {
                dimProduct *= shape[ii];
            }
            flatIdx += idx[i] * dimProduct;
        }

        return flatIdx;
    }
}

inline size_t numTensorElements(const std::vector< size_t >& shape)
{
    size_t size = 1;
    for (auto d : shape)
    {
        size *= d;
    }
    return size;
}

template< typename ScalarType >
ScalarType stringToScalar(const std::string& str)
{
    std::stringstream s(str);
    ScalarType scalar;
    s >> scalar;
    return scalar;
}


template< class T >
concept Arithmetic = std::is_arithmetic_v< T >;

template< Arithmetic ComponentType >
class Tensor
{
public:
    // Constructs a tensor with rank = 0 and zero-initializes the element.
    Tensor();

    // Constructs a tensor with arbitrary shape and zero-initializes all elements.
    Tensor(const std::vector< size_t >& shape);

    // Constructs a tensor with arbitrary shape and fills it with the specified value.
    explicit Tensor(const std::vector< size_t >& shape, const ComponentType& fillValue);

    // Copy-constructor.
    Tensor(const Tensor< ComponentType >& other);

    // Move-constructor.
    Tensor(Tensor< ComponentType >&& other) noexcept;

    // Copy-assignment
    Tensor&
    operator=(const Tensor< ComponentType >& other);

    // Move-assignment
    Tensor&
    operator=(Tensor< ComponentType >&& other) noexcept;

    // Destructor
    ~Tensor() = default;

    // Returns the rank of the tensor.
    [[nodiscard]] size_t rank() const;

    // Returns the shape of the tensor.
    [[nodiscard]] std::vector< size_t > shape() const;

    // Returns the number of elements of this tensor.
    [[nodiscard]] size_t numElements() const;

    // Element access function
    const ComponentType&
    operator()(const std::vector< size_t >& idx) const;

    // Element mutation function
    ComponentType&
    operator()(const std::vector< size_t >& idx);

private:

    std::vector< size_t > shape_;
    std::vector< ComponentType > data_;

};


template< Arithmetic ComponentType >
Tensor< ComponentType >::Tensor()
    : shape_(0), data_(1, 0)
{
}

template< Arithmetic ComponentType >
Tensor< ComponentType >::Tensor(const std::vector< size_t >& shape)
    : shape_(shape), data_(numTensorElements(shape), 0)
{
}

template< Arithmetic ComponentType >
Tensor< ComponentType >::Tensor(const std::vector< size_t >& shape, const ComponentType& fillValue)
    : shape_(shape), data_(numTensorElements(shape), fillValue)
{
}

// Copy-assignment
template< Arithmetic ComponentType >
Tensor< ComponentType >::Tensor(const Tensor< ComponentType >& other) = default;


// Move-constructor.
template< Arithmetic ComponentType >
Tensor< ComponentType >::Tensor(Tensor< ComponentType >&& other) noexcept
    : shape_(std::exchange(other.shape_, std::vector< size_t >())), data_(std::exchange(other.data_, {0}))
{
}

// Copy-assignment
template< Arithmetic ComponentType >
Tensor< ComponentType >& Tensor< ComponentType >::operator=(const Tensor< ComponentType >& other) = default;


// Move-assignment
template< Arithmetic ComponentType >
Tensor< ComponentType >& Tensor< ComponentType >::operator=(Tensor< ComponentType >&& other) noexcept

{
    shape_ = std::exchange(other.shape_, std::vector< size_t >());
    data_ = std::exchange(other.data_, {0});
    return *this;
}

template< Arithmetic ComponentType >
size_t
Tensor< ComponentType >::rank() const
{
    return shape_.size();
}

template< Arithmetic ComponentType >
std::vector< size_t >
Tensor< ComponentType >::shape() const
{
    return shape_;
}

template< Arithmetic ComponentType >
size_t
Tensor< ComponentType >::numElements() const
{
    return numTensorElements(shape_);
}

template< Arithmetic ComponentType >
const ComponentType&
Tensor< ComponentType >::operator()(const std::vector< size_t >& idx) const
{
    assert(idx.size() == rank());
    return data_[flatIdx(shape_, idx)];
}

template< Arithmetic ComponentType >
ComponentType&
Tensor< ComponentType >::operator()(const std::vector< size_t >& idx)
{
    assert(idx.size() == rank());
    return data_[flatIdx(shape_, idx)];
}


// Returns true if the shapes and all elements of both tensors are equal.
template< Arithmetic ComponentType >
bool operator==(const Tensor< ComponentType >& a, const Tensor< ComponentType >& b)
{

    if (a.shape() != b.shape())
    {
        return false;
    }

    size_t rank = a.rank();
    std::vector< size_t > shape = a.shape();
    size_t numElements = a.numElements();

    bool equal = true;

    if (rank == 0)
    {
        std::vector< size_t > idx(0);
        return a(idx) == b(idx);
    }
    else if (rank == 1)
    {
        for (size_t i = 0; i < shape[0]; i++)
        {
            std::vector< size_t > idx(1);
            idx[0] = i;
            equal &= a(idx) == b(idx);
        }
    }
    else
    {
        size_t cnt = 0;
        std::vector< size_t > idx(rank, 0);

        while (cnt < numElements)
        {
            for (size_t i = 0; i < shape[rank - 1]; i++)
            {
                equal &= a(idx) == b(idx);
                idx[rank - 1]++;
            }

            idx[rank - 1]++;

            for (size_t i = rank - 1; i > 0; i--)
            {
                if (idx[i] >= shape[i])
                {
                    idx[i] = 0;
                    idx[i - 1]++;
                }
            }

            cnt += shape[rank - 1];
        }
    }

    return equal;
}

// Pretty-prints the tensor to stdout.
// This is not necessary (and not covered by the tests) but nice to have, also for debugging (and for exercise of course...).
template< Arithmetic ComponentType >
std::ostream&
operator<<(std::ostream& out, const Tensor< ComponentType >& tensor)
{

    if (tensor.rank() == 0)
    {
        std::vector< size_t > idx(0);
        out << "() [" << tensor(idx) << "]\n";
    }
    else if (tensor.rank() == 1)
    {
        out << "(:) [";
        for (size_t i = 0; i < tensor.shape()[0] - 1; i++)
        {
            std::vector< size_t > idx(1);
            idx[0] = i;
            out << tensor(idx) << " ";
        }
        std::vector< size_t > idx(1);
        idx[0] = tensor.shape()[0] - 1;
        out << tensor(idx) << "]\n";
    }
    else
    {
        size_t cnt = 0;
        std::vector< size_t > idx(tensor.rank(), 0);

        while (cnt < tensor.numElements())
        {
            out << "(";
            for (size_t i = 0; i < tensor.rank() - 1; i++)
            {
                out << idx[i] << ", ";
            }
            out << ":) [";
            for (size_t i = 0; i < tensor.shape()[tensor.rank() - 1] - 1; i++)
            {
                out << tensor(idx) << " ";
                idx[tensor.rank() - 1]++;
            }

            out << tensor(idx) << "]\n";
            idx[tensor.rank() - 1]++;

            for (size_t i = tensor.rank() - 1; i > 0; i--)
            {
                if (idx[i] >= tensor.shape()[i])
                {
                    idx[i] = 0;
                    idx[i - 1]++;
                }
            }

            cnt += tensor.shape()[tensor.rank() - 1];
        }
    }

    return out;
}

// Reads a tensor from file.
template< Arithmetic ComponentType >
Tensor< ComponentType > readTensorFromFile(const std::string& filename)
{

    std::ifstream file;
    file.open(filename);

    if (!file.is_open())
    {
        std::cerr << "Could not open file." << std::endl;
        std::exit(1);
    }

    std::string line;
    std::getline(file, line);

    auto rank = stringToScalar< size_t >(line);

    std::vector< size_t > shape(rank);
    for (size_t i = 0; i < rank; i++)
    {
        std::getline(file, line);
        shape[i] = stringToScalar< size_t >(line);
    }

    Tensor< ComponentType > tensor(shape);

    if (rank == 0)
    {
        std::getline(file, line);
        tensor(shape) = stringToScalar< ComponentType >(line);
    }
    else
    {
        std::vector< size_t > idx(shape.size(), 0);
        size_t cnt = 0;
        while (cnt < tensor.numElements())
        {
            std::getline(file, line);
            tensor(idx) = stringToScalar< ComponentType >(line);

            idx[rank - 1]++;
            for (size_t i = rank - 1; i > 0; i--)
            {
                if (idx[i] >= shape[i])
                {
                    idx[i] = 0;
                    idx[i - 1]++;
                }
            }

            cnt++;
        }
    }

    file.close();
    return tensor;
}

// Writes a tensor to file.
template< Arithmetic ComponentType >
void writeTensorToFile(const Tensor< ComponentType >& tensor, const std::string& filename)
{

    std::ofstream file;
    file.open(filename);

    file << tensor.rank() << "\n";
    for (auto d : tensor.shape())
    {
        file << d << "\n";
    }

    if (tensor.rank() == 0)
    {
        file << tensor({}) << "\n";
    }
    else
    {
        std::vector< size_t > idx(tensor.shape().size(), 0);
        size_t cnt = 0;
        while (cnt < tensor.numElements())
        {
            file << tensor(idx) << "\n";

            idx[tensor.rank() - 1]++;
            for (size_t i = tensor.rank() - 1; i > 0; i--)
            {
                if (idx[i] >= tensor.shape()[i])
                {
                    idx[i] = 0;
                    idx[i - 1]++;
                }
            }

            cnt++;
        }
    }

    file.close();
}
