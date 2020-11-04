#include <torch/extension.h>

#include <iostream>
#include <vector>
#include <time.h>
#include <stdlib.h>
#include <iomanip>
#include <random>
#include <bitset>
#include <math.h>
#include <algorithm>

// ????????????????????????
float toInt(int var, const std::size_t width, const bool isSigned)
{
    std::uint64_t repeatingSignMask = std::numeric_limits<std::uint64_t>::max();
    for (std::size_t i = 0; i < width; i++)
    {
        repeatingSignMask &= ~(1ul << i);
    } // ex repeatingSignMask: 0b11111000

    // if leftmost relevant bit is 1 (if signed)
    const bool isNegative =  (var >> (width - 1) & 1);
    if (isSigned && isNegative)
    {
        var |= repeatingSignMask; // repeat 1 leftwards
    }
    else
    {
        var &= ~repeatingSignMask; // repeat 0 leftwards
    }

    return static_cast<float>(static_cast<uint64_t>(var));
}


float toBinary(int var, const bool isSigned)
{
    return var < 0 ? -1*isSigned : 1;
}


float toFixedPoint(float var, const std::size_t width, const std::size_t nbDigits)
{
    float precision = 1.0 / (1u << nbDigits); // Smallest increment with the set number of bits
    std::size_t wholeWidth = width - nbDigits; // width of the whole part of the number (int(var))
    
    int maxInt = 1u << (wholeWidth-1); // Max value possible of the whole part with the number of bits specified, assuming signed notation
    std::clamp(var, -1*maxInt, maxInt - precision) // Clamping variable to calculated values

    return var - fmod(var, precision);
}


std::uint64_t encodeFixedPoint(float var, const std::size_t width, const std::size_t nbDigits)
{
    float precision = 1.0 / (1u << nbDigits); // Smallest increment with the set number of bits
    std::size_t wholeWidth = width - nbDigits; // width of the whole part of the number (int(var))
    
    int maxInt = 1u << (wholeWidth-1); // Max value possible of the whole part with the number of bits specified, assuming signed notation
    std::clamp(var, -1*maxInt, maxInt - precision) // Clamping variable to calculated values

    int whole = static_cast<int>(var);
    int dec = static_cast<uint>((var - whole) / precision);
    return static_cast<std::uint64_t>((whole << nbDigits) ^ dec);
    // Example: 000...00010100 ^ 000...00011 = 000...00010111, with width = 5, nbDigits = 2, gives a result of -2.25
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("toInt", &toInt, "CPPperturb");
}
