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


namespace customFunc
{
    float clamp(float v, float lo, float hi)
    {
        assert( !(hi < lo) );
        return (v < lo) ? lo : (hi < v) ? hi : v; 
    }
}



//*************************************
// Binary Representation
//*************************************

// Encode
std::uint8_t encodeBinary(const float var, const bool isSigned = true)
{
    return var > 0;
}

torch::Tensor encodeTenBinary(const torch::Tensor& ten, const bool isSigned = true)
{
    at::Tensor boolTen = ten > 0;
    return boolTen.to(torch::kUInt8);
}

// Decode
float decodeBinary(const std::uint8_t var, const bool isSigned = true)
{
    return var == 0 ? -1*isSigned : 1;
}

torch::Tensor decodeTenBinary(const torch::Tensor& ten, const bool isSigned = true)
{
    at::Tensor floatTen = ten.to(torch::kFloat32);
    if(isSigned)
    {
        return floatTen*2 - 1; 
    }
    else
    {
        return floatTen;
    }
}

// Quantize
float quantizeBinary(const float var, const bool isSigned = true)
{
    return var <= 0 ? -1*isSigned : 1;
}

torch::Tensor quantizeTenBinary(const torch::Tensor& ten, const bool isSigned = true)
{
    at::Tensor boolTen = ten > 0;
    at::Tensor floatTen = boolTen.to(torch::kFloat32);
    if(isSigned)
    {
        return floatTen*2 - 1; 
    }
    else
    {
        return floatTen;
    }
}

//*************************************
// Integer Representation
//*************************************

// Encode
std::uint8_t encodeInt(const float var, const std::size_t width, const bool isSigned)
{
    // Finding representation boundaries
    int low = -(1u << (width - 1)) * isSigned;   
    int high = low + (1u << width) - 1;
    
    // Clamping and rounding, then converting to uint8
    return static_cast<std::uint8_t>(std::round(customFunc::clamp(var, low, high)));
}

torch::Tensor encodeTenInt(const torch::Tensor& ten, const std::size_t width, const bool isSigned)
{
    // Finding representation boundaries
    int low = -(1u << (width - 1)) * isSigned;   
    int high = low + (1u << width) - 1;
    
    // Clamping and rounding, then converting to uint8
    return ten.clamp(low, high).round().to(torch::kUInt8);
    
}

// Decode
float decodeInt(std::uint8_t var, const std::size_t width, const bool isSigned)
{
    std::uint8_t repeatingSignMask = std::numeric_limits<std::uint8_t>::max();
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

    return static_cast<float>(static_cast<int8_t>(var));
}

torch::Tensor decodeTenInt(const torch::Tensor& ten, const std::size_t width, const bool isSigned)
{
    
    std::uint8_t repeatingSignMask = std::numeric_limits<std::uint8_t>::max();
    for (std::size_t i = 0; i < width; i++)
    {
        repeatingSignMask &= ~(1ul << i);
    } // ex repeatingSignMask: 0b11111000


    torch::Tensor floatTen = ten.to(torch::kFloat32).flatten();
    torch::Tensor flat = ten.flatten().to(torch::kInt32);
    auto access = flat.accessor<int,1>();
    for(int i = 0; i < flat.size(0); i++)
    {
        std::uint8_t var = static_cast<std::uint8_t>(access[i]);
        const bool isNegative = (var >> (width - 1) & 1);
        if (isSigned && isNegative)
        {
            var |= repeatingSignMask; // repeat 1 leftwards
        }
        else
        {
            var &= ~repeatingSignMask; // repeat 0 leftwards
        }
        
        floatTen[i] = static_cast<float>(static_cast<int8_t>(var));
    }

    return floatTen.reshape_as(ten);
}


// Quantize
float quantizeInt(const float var, const std::size_t width, const bool isSigned)
{
    // Finding representation boundaries
    int low = -(1u << (width - 1)) * isSigned;   
    int high = low + (1u << width) - 1;
    
    // Clamping and rounding
    return std::round(customFunc::clamp(var, low, high));
}

at::Tensor quantizeTenInt(const at::Tensor ten, const std::size_t width, const bool isSigned)
{
    // Finding representation boundaries
    int low = -(1u << (width - 1)) * isSigned;   
    int high = low + (1u << width) - 1;
    
    // Clamping and rounding
    return ten.clamp(low, high).round();
}


//*************************************
// Fixed-Point Representation
//*************************************

// Encode
std::uint8_t encodeFixedPoint(const float var, const std::size_t width, const std::size_t nbDigits)
{
    float precision = 1.0 / (1u << nbDigits); // Smallest increment with the set number of bits
    std::size_t wholeWidth = width - nbDigits; // width of the whole part of the number (int(var))

    int maxInt = 1u << (wholeWidth); // Max value possible of the whole part with the number of bits specified, assuming unsigned notation
    float maxVal = maxInt/2; // Max value possible of the whole part with the number of bits specified, assuming signed notation
    float clamped = customFunc::clamp(var, -1*maxVal, maxVal - precision); // Clamping variable to calculated values

    clamped *= 1u << nbDigits;
    clamped = std::round(clamped);
    return static_cast<std::uint8_t>(clamped);
}

torch::Tensor encodeTenFixedPoint(const torch::Tensor& ten, const std::size_t width, const std::size_t nbDigits)
{
    float precision = 1.0 / (1u << nbDigits); // Smallest increment with the set number of bits
    std::size_t wholeWidth = width - nbDigits; // width of the whole part of the number (int(var))

    int maxInt = 1u << (wholeWidth); // Max value possible of the whole part with the number of bits specified, assuming unsigned notation
    float maxVal = maxInt/2.f; // Max value possible of the whole part with the number of bits specified, assuming signed notation
    torch::Tensor clamped = ten.clamp(-maxVal, maxVal - precision); // Clamping tensor variable to calculated values

    //std::cerr << precision << '\n';
    //std::cerr << maxVal << '\n';
    //std::cerr << maxInt << '\n';

    unsigned int scale = 1 << nbDigits;
    clamped = clamped * static_cast<int>(scale);
    return clamped.round().to(torch::kUInt8);
}

// Decode
float decodeFixedPoint(std::uint8_t var, const std::size_t width, const std::size_t nbDigits)
{
    std::int8_t varInt = static_cast<std::int8_t>(var);
    float precision = 1.0 / (1u << nbDigits);
    return precision*varInt;
}

torch::Tensor decodeTenFixedPoint(const torch::Tensor& ten, const std::size_t width, const std::size_t nbDigits)
{
    torch::Tensor intTen = ten.to(torch::kInt8).to(torch::kFloat32);
    float precision = 1.0 / (1u << nbDigits);
    return precision*intTen;
}

// Quantize
float quantizeFixedPoint(float var, const std::size_t width, const std::size_t nbDigits)
{
    float precision = 1.0 / (1u << nbDigits); // Smallest increment with the set number of bits
    std::size_t wholeWidth = width - nbDigits; // width of the whole part of the number (int(var))
    
    int maxInt = 1u << (wholeWidth-1); // Max value possible of the whole part with the number of bits specified, assuming signed notation

    float clamped = customFunc::clamp(var, -1*maxInt, maxInt - precision); // Clamping variable to calculated values

    int f = (1u << nbDigits);
    clamped *= f;
    clamped = std::round(clamped);
    return clamped /= f;
}

torch::Tensor quantizeTenFixedPoint(const torch::Tensor ten, const std::size_t width, const std::size_t nbDigits)
{
    float precision = 1.0 / (1u << nbDigits); // Smallest increment with the set number of bits
    std::size_t wholeWidth = width - nbDigits; // width of the whole part of the number (int(var))
    
    int maxInt = 1u << (wholeWidth-1); // Max value possible of the whole part with the number of bits specified, assuming signed notation

    torch::Tensor clamped = ten.clamp(-1*maxInt, maxInt - precision); // Clamping variable to calculated values

    int f = (1u << nbDigits);
    clamped *= f;
    clamped = clamped.round();
    return clamped /= f;
}


// Binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("encodeBinary", &encodeBinary, "CPPperturb");
    m.def("encodeTenBinary", &encodeTenBinary, "CPPperturb");
    m.def("decodeBinary", &decodeBinary, "CPPperturb");
    m.def("decodeTenBinary", &decodeTenBinary, "CPPperturb");
    m.def("quantizeBinary", &quantizeBinary, "CPPperturb");
    m.def("quantizeTenBinary", &quantizeTenBinary, "CPPperturb");
    
    m.def("encodeInt", &encodeInt, "CPPperturb");
    m.def("encodeTenInt", &encodeTenInt, "CPPperturb");
    m.def("decodeInt", &decodeInt, "CPPperturb");
    m.def("decodeTenInt", &decodeTenInt, "CPPperturb");
    m.def("quantizeInt", &quantizeInt, "CPPperturb");
    m.def("quantizeTenInt", &quantizeTenInt, "CPPperturb");
    
    m.def("encodeFixedPoint", &encodeFixedPoint, "CPPperturb");
    m.def("encodeTenFixedPoint", &encodeTenFixedPoint, "CPPperturb");
    m.def("decodeFixedPoint", &decodeFixedPoint, "CPPperturb");
    m.def("decodeTenFixedPoint", &decodeTenFixedPoint, "CPPperturb");
    m.def("quantizeFixedPoint", &quantizeFixedPoint, "CPPperturb");
    m.def("quantizeTenFixedPoint", &quantizeTenFixedPoint, "CPPperturb");
}
