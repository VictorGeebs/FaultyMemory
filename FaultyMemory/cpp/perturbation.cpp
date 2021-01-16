#include <torch/extension.h>

#include <iostream>
#include <vector>
#include <time.h>
#include <stdlib.h>
#include <iomanip>
#include <random>
#include <bitset>
#include <algorithm>

// TODO : perturbator : generer un tableau de taille du tenseur + 1 dimension ajoutée avec le sample booleen
// géré en C : concaténer la dim +1 pour obtenir le masque de perturbation
// géré en pytorch : xor, or, and

namespace randomGens
{
    std::random_device rd;
    std::mt19937 gen(randomGens::rd());
}


void signInvert(torch::Tensor& param, float p)
{
    std::bernoulli_distribution d(p);
    
    auto ten = param.accessor<float,1>();
    for(int i = 0; i < ten.size(0); i++)
    {
        param[i] = ten[i] * ((d(randomGens::gen) * -2) + 1);
    }
}


std::uint8_t generateRandomBitArray(const std::size_t width, const float p)
{
    std::bernoulli_distribution d(p);

    std::uint8_t perturbMask = std::numeric_limits<std::uint64_t>::min();
    for (std::size_t i = 0; i < width; i++)
    {
        perturbMask |= (1u << i)*d(randomGens::gen);
    } 
    return perturbMask;
}

at::Tensor generateTensorMask(const torch::Tensor& ten, const std::size_t width, const float p)
{
    std::vector<int64_t> shape_vec;
    int length = ten.sizes().size();
    shape_vec.resize(length+1);

    for(int i = 0; i < length; i++)
    {
        int64_t x = ten.sizes()[i];
        shape_vec[i] = x;
    }
    shape_vec[length] = width;
    c10::ArrayRef<int64_t> arr = c10::ArrayRef<int64_t>(shape_vec);
    
    return at::zeros(arr).bernoulli(p);
}

void applyPerturbMask(torch::Tensor& param, const std::size_t width, const float p)
{
    auto ten = param.accessor<float,1>();
    for(int i = 0; i < ten.size(0); i++)
    {
        param[i] = static_cast<uint8_t>(ten[i]) ^ generateRandomBitArray(width, p);
    }
}








PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("generateTensorMask", &generateTensorMask, "CPPperturb");
    m.def("signInvert", &signInvert, "CPPperturb");
    m.def("generateRandomBitArray", &generateRandomBitArray, "CPPperturb");
    m.def("applyPerturbMask", &applyPerturbMask, "CPPperturb");
}
