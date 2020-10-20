#include <torch/extension.h>

#include <iostream>
#include <vector>
#include <time.h>
#include <stdlib.h>
#include <iomanip>
#include <random>

namespace randomGens
{
    std::random_device rd;
    std::mt19937 gen(randomGens::rd());
}

void perturb(torch::Tensor& param, float p)
{
    std::bernoulli_distribution d(p);
    
    auto ten = param.accessor<float,1>();
    for(int i = 0; i < ten.size(0); i++)
    {
        param[i] = ten[i] * ((d(randomGens::gen) * -2) + 1);
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("perturb", &perturb, "CPPperturb");
}
