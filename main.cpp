#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

// taken from
// https://github.com/pytorch/pytorch/blob/main/c10/util/Float8_e4m3fnuz.h
inline float fp32_from_bits(uint32_t w)
{
    union
    {
        uint32_t as_bits;
        float as_value;
    } fp32 = {w};
    return fp32.as_value;
}

inline uint32_t fp32_to_bits(float f)
{
    union
    {
        float as_value;
        uint32_t as_bits;
    } fp32 = {f};
    return fp32.as_bits;
}

// taken from
// https://github.com/pytorch/pytorch/blob/main/c10/util/Float8_e4m3fnuz.h
inline uint8_t fp8e4m3fnuz_from_fp32_value(float f)
{
    /*
     * Binary representation of 256.0f, which is the first value not representable
     * (i.e. the first value which would overflow in to the sign bit, resulting in
     * a NaN) in fp8e4m3fnuz range:
     * 1 0000 000 - fp8e4m3fnuz
     * 0 10000111 00000000000000000000000 - fp32
     */
    constexpr uint32_t fnuz_max = UINT32_C(0x87) << 23;

    /*
     * A mask for converting fp32 numbers lower than fp8e4m3fnuz normal range
     * into denormalized representation.
     * magic number: ((127 - 8) + (23 - 3) + 1)
     */
    constexpr uint32_t denorm_mask = UINT32_C(0x8C) << 23;

    uint32_t f_bits = fp32_to_bits(f);

    uint32_t result = 0u;

    /*
     * Extract the sign of the input number into the high bit of the 32-bit word:
     *
     *      +---+----------------------------------+
     *      | S |0000000 00000000 00000000 00000000|
     *      +---+----------------------------------+
     * Bits  31                 0-31
     */
    const uint32_t sign = f_bits & UINT32_C(0x80000000);

    /*
     * Set sign bit to 0
     */
    f_bits ^= sign;

    if(f_bits >= fnuz_max)
    {
        // NaN -- sign bit set to 1, rest 0s.
        return 0x80;
    }

    if(f_bits < (UINT32_C(0x78) << 23) /* 2^-7 in float32 */)
    {
        // Input exponent is less than -7, the smallest e4m3fnuz exponent, so the
        // number will become subnormal.
        f_bits = fp32_to_bits(fp32_from_bits(f_bits) + fp32_from_bits(denorm_mask));
        result = static_cast<uint8_t>(f_bits - denorm_mask);
        if(result == 0)
        {
            // fnuz types don't have negative zero.
            return 0;
        }
    }
    else
    {
        // resulting mantissa is odd
        uint8_t mant_odd = (f_bits >> 20) & 1;

        // update exponent, rounding bias part 1
        f_bits += ((uint32_t)(8 - 127) << 23) + 0x7FFFF;

        // rounding bias part 2
        f_bits += mant_odd;

        // take the bits!
        result = static_cast<uint8_t>(f_bits >> 20);
    }

    result |= sign >> 24;

    return result;
}

// taken from
// https://github.com/pytorch/pytorch/blob/main/c10/util/Float8_e4m3fn.h
/*
 * Convert a 32-bit floating-point number in IEEE single-precision format to a
 * 8-bit floating-point number in fp8 E4M3FN format, in bit representation.
 */
inline uint8_t fp8e4m3fn_from_fp32_value(float f)
{
    /*
     * Binary representation of 480.0f, which is the first value
     * not representable in fp8e4m3fn range:
     * 0 1111 111 - fp8e4m3fn
     * 0 10000111 11100000000000000000000 - fp32
     */
    constexpr uint32_t fp8_max = UINT32_C(1087) << 20;

    /*
     * A mask for converting fp32 numbers lower than fp8e4m3fn normal range
     * into denorm representation
     * magic number: ((127 - 7) + (23 - 3) + 1)
     */
    constexpr uint32_t denorm_mask = UINT32_C(141) << 23;

    uint32_t f_bits = fp32_to_bits(f);

    uint8_t result = 0u;

    /*
     * Extract the sign of the input number into the high bit of the 32-bit word:
     *
     *      +---+----------------------------------+
     *      | S |0000000 00000000 00000000 00000000|
     *      +---+----------------------------------+
     * Bits  31                 0-31
     */
    const uint32_t sign = f_bits & UINT32_C(0x80000000);

    /*
     * Set sign bit to 0
     */
    f_bits ^= sign;

    if(f_bits >= fp8_max)
    {
        // NaN - all exponent and mantissa bits set to 1
        result = 0x7f;
    }
    else
    {
        if(f_bits < (UINT32_C(121) << 23))
        {
            // Input number is smaller than 2^(-6), which is the smallest
            // fp8e4m3fn normal number
            f_bits = fp32_to_bits(fp32_from_bits(f_bits) + fp32_from_bits(denorm_mask));
            result = static_cast<uint8_t>(f_bits - denorm_mask);
        }
        else
        {
            // resulting mantissa is odd
            uint8_t mant_odd = (f_bits >> 20) & 1;

            // update exponent, rounding bias part 1
            f_bits += ((uint32_t)(7 - 127) << 23) + 0x7FFFF;

            // rounding bias part 2
            f_bits += mant_odd;

            // take the bits!
            result = static_cast<uint8_t>(f_bits >> 20);
        }
    }

    result |= static_cast<uint8_t>(sign >> 24);
    return result;
}

// taken from https://github.com/pytorch/pytorch/blob/main/c10/util/Float8_e5m2.h
/*
 * Convert a 32-bit floating-point number in IEEE single-precision format to a
 * 8-bit floating-point number in fp8 E5M2 format, in bit representation.
 */
inline uint8_t fp8e5m2_from_fp32_value(float f)
{
    /*
     * Binary representation of fp32 infinity
     * 0 11111111 00000000000000000000000
     */
    constexpr uint32_t fp32_inf = UINT32_C(255) << 23;

    /*
     * Binary representation of 65536.0f, which is the first value
     * not representable in fp8e5m2 range:
     * 0 11111 00 - fp8e5m2
     * 0 10001111 00000000000000000000000 - fp32
     */
    constexpr uint32_t fp8_max = UINT32_C(143) << 23;

    /*
     * A mask for converting fp32 numbers lower than fp8e5m2 normal range
     * into denorm representation
     * magic number: ((127 - 15) + (23 - 2) + 1)
     */
    constexpr uint32_t denorm_mask = UINT32_C(134) << 23;

    uint32_t f_bits = fp32_to_bits(f);
    uint8_t result  = 0u;

    /*
     * Extract the sign of the input number into the high bit of the 32-bit word:
     *
     *      +---+----------------------------------+
     *      | S |0000000 00000000 00000000 00000000|
     *      +---+----------------------------------+
     * Bits  31                 0-31
     */
    const uint32_t sign = f_bits & UINT32_C(0x80000000);

    /*
     * Set sign bit to 0
     */
    f_bits ^= sign;

    if(f_bits >= fp8_max)
    {
        // NaN - all exponent and mantissa bits set to 1
        result = f_bits > fp32_inf ? UINT8_C(0x7F) : UINT8_C(0x7C);
    }
    else
    {
        if(f_bits < (UINT32_C(113) << 23))
        {
            // Input number is smaller than 2^(-14), which is the smallest
            // fp8e5m2 normal number
            f_bits = fp32_to_bits(fp32_from_bits(f_bits) + fp32_from_bits(denorm_mask));
            result = static_cast<uint8_t>(f_bits - denorm_mask);
        }
        else
        {
            // resulting mantissa is odd
            uint32_t mant_odd = (f_bits >> 21) & 1;

            // update exponent, rounding bias part 1
            f_bits += ((uint32_t)(15 - 127) << 23) + 0xFFFFF;

            // rounding bias part 2
            f_bits += mant_odd;

            // take the bits!
            result = static_cast<uint8_t>(f_bits >> 21);
        }
    }

    result |= static_cast<uint8_t>(sign >> 24);
    return result;
}

// taken from https://github.com/pytorch/pytorch/blob/main/c10/util/Float8_e5m2fnuz.h
/*
 * Convert a 32-bit floating-point number in IEEE single-precision format to a
 * 8-bit floating-point number in fp8 E5M2 format, in bit representation.
 */
inline uint8_t fp8e5m2fnuz_from_fp32_value(float f)
{
    /*
     * Binary representation of 65536.0f, which is the first value not
     * representable (i.e. the first value which would overflow in to the sign
     * bit, resulting in a NaN) in fp8e4m3fnuz range:
     * 1 00000 00 - fp8e5m2fnuz
     * 0 10001111 00000000000000000000000 - fp32
     */
    constexpr uint32_t fnuz_max = UINT32_C(0x8F) << 23;

    /*
     * A mask for converting fp32 numbers lower than fp8e5m2fnuz normal range
     * into denormalized representation.
     * magic number: ((127 - 16) + (23 - 2) + 1)
     */
    constexpr uint32_t denorm_mask = UINT32_C(0x85) << 23;

    uint32_t f_bits = fp32_to_bits(f);

    uint32_t result = 0u;

    /*
     * Extract the sign of the input number into the high bit of the 32-bit word:
     *
     *      +---+----------------------------------+
     *      | S |0000000 00000000 00000000 00000000|
     *      +---+----------------------------------+
     * Bits  31                 0-31
     */
    const uint32_t sign = f_bits & UINT32_C(0x80000000);

    /*
     * Set sign bit to 0
     */
    f_bits ^= sign;

    if(f_bits >= fnuz_max)
    {
        // NaN -- sign bit set to 1, rest 0s
        return 0x80;
    }

    if(f_bits < (UINT32_C(0x70) << 23) /* 2^-15 in float32 */)
    {
        // Input exponent is less than -15, the smallest e5m2fnuz exponent, so the
        // number will become subnormal.
        f_bits = fp32_to_bits(fp32_from_bits(f_bits) + fp32_from_bits(denorm_mask));
        result = static_cast<uint8_t>(f_bits - denorm_mask);
        if(result == 0)
        {
            // fnuz types don't have negative zero.
            return 0;
        }
    }
    else
    {
        // resulting mantissa is odd
        uint8_t mant_odd = (f_bits >> 21) & 1;

        // update exponent, rounding bias part 1
        f_bits += ((uint32_t)(16 - 127) << 23) + 0xFFFFF;

        // rounding bias part 2
        f_bits += mant_odd;

        // take the bits!
        result = static_cast<uint8_t>(f_bits >> 21);
    }

    result |= sign >> 24;
    return result;
}

#define MIGRAPHX_CONST_FOLD(x) (__builtin_constant_p(x) ? (x) : (x))

template <typename To, typename From>
inline constexpr To bit_cast(From fr) noexcept
{
    static_assert(sizeof(To) == sizeof(From));
#if defined(__GNUC__) and !defined(__clang__)
    return MIGRAPHX_CONST_FOLD(*reinterpret_cast<To*>(&fr));
#else
    return __builtin_bit_cast(To, fr);
#endif
}

// MIGraphX FP8 implementation
template <typename T, int Wm, int We, bool NegativeZeroNan, bool Clip = true>
uint8_t cast_to_f8(T f_x, bool stoch = false, uint32_t rng = 0)
{
    constexpr bool is_float = std::is_same<T, float>::value;
    // half is not supported for now
    constexpr bool is_half = false;
    static_assert(Wm + We == 7, "Wm+We==7");
    static_assert(is_float or is_half, "Only float can be cast to f8");

    const uint32_t mfmt = (sizeof(T) == 4) ? 23 : 10;
    typename std::conditional<sizeof(T) == 2, uint16_t, uint32_t>::type x;

    if constexpr(sizeof(T) == 4)
        x = bit_cast<uint32_t>(f_x);
    else
        x = bit_cast<uint16_t>(f_x);

    uint32_t head     = 0;
    uint32_t mantissa = 0;
    int exponent      = 0;
    uint32_t bias     = 0;
    uint32_t sign     = 0;
    if constexpr(sizeof(T) == 4)
    {
        head     = x & 0xFF800000;
        mantissa = x & 0x7FFFFF;
        exponent = (head >> 23) & 0xFF;
        sign     = head >> 31;
        bias     = 127;
    }
    else
    {
        head     = x & 0xFC00;
        mantissa = x & 0x3FF;
        exponent = (head >> 10) & 0x1F;
        sign     = head >> 15;
        bias     = 15;
    }

    uint32_t signed_inf      = (sign << 7) + (((1 << We) - 1) << Wm);
    uint32_t signed_all_ones = (sign << 7) + ((((1 << We) - 1) << Wm) + ((1 << Wm) - 1));

    // Calcualte maximum singed value FLT_MAX, FLT_MIN
    uint32_t signed_max = signed_all_ones;
    if(not NegativeZeroNan)
        signed_max = (Wm == 2) ? (signed_max - 4) : (signed_max - 1);

    // Deal with inf and NaNs
    if(NegativeZeroNan) // For the FNUZ cases, it is simple just return NaNs
    {
        if((sizeof(T) == 4 and ((x & 0x7F800000) == 0x7F800000)) or
           (sizeof(T) == 2 and ((x & 0x7C00) == 0x7C00)))
            return 0x80;
    }
    else
    {
        // calculate most common NaN mantissa for FP8, which is all Ones in binary
        uint32_t nan_mantissa = 1;
        for(auto i = 1; i < Wm; ++i)
        {
            nan_mantissa |= (nan_mantissa << 1);
        }
        if((sizeof(T) == 4 and ((x & 0x7F800000) == 0x7F800000)) or
           (sizeof(T) == 2 and ((x & 0x7C00) == 0x7C00)))
        {
            // infinity
            if(mantissa == 0)
            {
                if(sign == 0)
                    return (Wm == 2) ? 0x7B : 0x7E;
                else
                    return (Wm == 2) ? 0xFB : 0xFE;
            }
            else // NaNs
                return signed_inf + nan_mantissa;
        }
    }
    // handle positive zero
    if(x == 0)
        return 0;
    // handle negative zero
    else if((sizeof(T) == 4 and x == 0x80000000) or (sizeof(T) == 2 and x == 0x8000))
    {
        return NegativeZeroNan ? 0 : 0x80; // For FNUZ types neg zero is just positive zero
    }

    /* First need to check if it is normal or denorm as there is a difference of
    implict 1 Then need to adjust the exponent to align with the F8 exponent, in
    the meanwhile, shift The mantissa. Then for stochastic rounding, add rng to
    mantissa and truncate. And for RNE, no need to add rng. Then probably need to
    check whether there is carry and adjust exponent and mantissa again*/

    // For IEEE bias mode, the bias is 2^(k-1) -1 where k is the width of exponent
    // bits
    const int f8_bias                  = (1 << (We - 1u)) - 1 + (NegativeZeroNan ? 1 : 0);
    const int f8_denormal_act_exponent = 1 - f8_bias; // actual exponent of f8 denormal
    /* act_exponent is the actual exponent of fp32/fp16 (after subtracting bias)
    f8_exponent is the converted f8 exponent with bias encoding
    exponent_diff is the diff between fp32/fp16 exponent and f8 exponent,
    the difference needs to be adjusted and mantissa shifted*/
    int act_exponent  = 0;
    int f8_exponent   = 0;
    int exponent_diff = 0;

    if(exponent == 0 and mantissa != 0)
    { // fp32/fp16 is in denormal.
        /* fp32 denormal is below 2^-127 so it is usually not a concern here, we
        mostly concern fp16 here. In this case, f8 is usually in denormal. But there
        could be exceptions. fp16 denormal has exponent bias 15 while bf8 with FNUZ
        has exponent bias 16. It means that there are some numbers in fp16 denormal
        but they are bf8 (FNUZ) normals - smallest bf8 (FNUZ) normal is 2^-15.
        fp16 numbers where exponent==0 (actual exponent -14) and highest bit of
        mantissa is 1 are bf8 (FNUZ) normal. In this case, the fp16 mantissa should
        be shift left by 1  */
        act_exponent  = 1 - bias;
        exponent_diff = f8_denormal_act_exponent -
                        act_exponent; // actual exponent is exponent-bias+1 as it is denormal
    }
    else
    { // fp32/fp16 is normal with implicit 1
        act_exponent = exponent - bias;
        if(act_exponent <= f8_denormal_act_exponent)
        {
            /* This is the case where fp32/fp16 is normal but it is in f8 denormal
            range. For example fp8 fnuz mode, denormal exponent is -7, but if the
            fp32/fp16 actual exponent is -7, it is actually larger due to the implict
            1, Therefore it needs to be adjust to -6 and mantissa shift right by 1. So
            for fp32/fp16, exponent -8 is the cut point to convert to fp8 fnuz */
            exponent_diff = f8_denormal_act_exponent - act_exponent;
        }
        else
        {                      // both fp32/fp16 and f8 are in normal range
            exponent_diff = 0; // exponent_diff=0 does not mean there is no difference
                               // for this case,
            // act_exponent could be larger. Just that it does not need shift mantissa
        }
        mantissa += (1u << mfmt); // Add the implicit 1 into mantissa
    }

    bool midpoint = (mantissa & ((1 << (mfmt - Wm + exponent_diff)) - 1)) ==
                    (1 << (mfmt - Wm + exponent_diff - 1));
    /* This part is a bit tricky. The judgment of whether it is a tie needs to be
    done before we shift right as shift right could rip off some residual part and
    make something not midpoint look like midpoint. For example, the fp16 number
    0x1002 (0 00100 0000000010), it is larger than midpoint, but after shift right
    by 4 bits, it would look like midpoint.
    */

    if(exponent_diff > 0)
        mantissa >>= exponent_diff;
    else if(exponent_diff == -1)
        mantissa <<= -exponent_diff;
    bool implicit_one = mantissa & (1 << mfmt);
    // if there is no implict 1, it  means the f8 is denormal and need to adjust
    // to denorm exponent
    f8_exponent =
        (act_exponent + exponent_diff) /*actual f8 exponent*/ + f8_bias - (implicit_one ? 0 : 1);

    // Now we have the exponent and mantissa adjusted
    uint32_t drop_mask = (1u << (mfmt - Wm)) - 1;
    bool odd =
        mantissa & (1u << (mfmt - Wm)); // if the least significant bit that is not truncated is 1
    mantissa += (stoch ? rng : (midpoint ? (odd ? mantissa : mantissa - 1) : mantissa)) & drop_mask;

    // Now we deal with overflow
    if(f8_exponent == 0 and ((1 << mfmt) & mantissa))
    {
        f8_exponent = 1; // denormal overflow to become normal, promote exponent
    }
    else if((1 << (mfmt + 1)) & mantissa)
    {
        mantissa >>= 1;
        f8_exponent++;
    }

    mantissa >>= (mfmt - Wm);

    // above range: quantize to maximum possible float of the same sign
    // for e5m2 case, max_exp is 14, since exp = 15 is reserved for Infs and Nans
    const int max_exp = (1 << We) - ((NegativeZeroNan or Wm == 3) ? 1 : 2);
    if(f8_exponent > max_exp)
    {
        if(Clip)
            return signed_max;
        else
        {
            // https://onnx.ai/onnx/technical/float8.html#cast
            if(NegativeZeroNan)
                return 0x80;
            else
                return (Wm == 2) ? signed_inf : signed_all_ones;
        }
    }

    if(f8_exponent == 0 and mantissa == 0)
        return NegativeZeroNan ? 0 : (sign << 7);
    mantissa &= (1 << Wm) - 1;
    return (sign << 7) | (f8_exponent << Wm) | mantissa;
}

int main(int argc, char** argv) { return 0; }
