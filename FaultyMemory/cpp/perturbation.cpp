#include <torch/extension.h>

#include <iostream>
#include <vector>
#include <time.h>
#include <stdlib.h>
#include <iomanip>
#include <random>
#include <bitset>


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


std::uint64_t generateRandomBitArray(const std::size_t width, const float p)
{
    std::bernoulli_distribution d(p);

    std::uint64_t perturbMask = std::numeric_limits<std::uint64_t>::min();
    for (std::size_t i = 0; i < width; i++)
    {
        perturbMask |= (1u << i)*d(randomGens::gen);
    } 
    std::cerr << std::bitset<8>(perturbMask) << '\n';
    return perturbMask;
}


void applyPerturbMask(torch::Tensor& param, const std::size_t width, const float p)
{
    auto ten = param.accessor<float,1>();
    for(int i = 0; i < ten.size(0); i++)
    {
        param[i] = static_cast<uint64_t>(ten[i]) ^ generateRandomBitArray(width, p);
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("signInvert", &signInvert, "CPPperturb");
    m.def("generateRandomBitArray", &generateRandomBitArray, "CPPperturb");
    m.def("applyPerturbMask", &applyPerturbMask, "CPPperturb");
}
