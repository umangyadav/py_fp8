import numpy as np
from enum import Enum, IntEnum


class Rounding(Enum):
    standard = 0
    stochastic = 1


class Sign(IntEnum):
    positive = 0
    negative = 1


mantissa_map_m3 = {0: 0, 1: 0.125, 2: 0.25,
                   3: 0.375, 4: 0.5, 5: 0.625, 6: 0.75, 7: 0.875}
mantissa_map_m2 = {0: 0, 1: 0.25, 2: 0.5, 3: 0.75}


class Float8:
    def __init__(self, wm, we, fnuz, clip, rounding):
        self.wm: int = wm
        self.we: int = we
        self.fnuz: bool = fnuz
        self.clip: bool = clip
        self.rounding: Rounding = rounding
        self.exp_bias: int = (2 ** (self.we-1)) - (0 if self.fnuz else 1)
        assert (wm == 2 or wm == 3)
        assert (we == 4 or we == 5)
        assert (wm + we == 7)

    def construct_fp8(self, mantissa: int, exponent: int, sign: Sign):
        fmantissa = 1.0
        mantissa_shift = 0
        if self.wm == 2:
            assert (mantissa >= 0 and mantissa <= 3)
            mantissa_shift = mantissa_map_m2[mantissa]
            fmantissa += mantissa_shift
        elif self.wm == 3:
            assert (mantissa >= 0 and mantissa <= 7)
            mantissa_shift = mantissa_map_m3[mantissa]
            fmantissa += mantissa_shift

        if self.we == 5:
            assert (exponent >= 0 and exponent <= 31)
        elif self.we == 4:
            assert (exponent >= 0 and exponent <= 15)
        # need to handle exponent = 0 case
        if exponent == 0:
            return ((-1) ** sign) * (2 ** (-(self.exp_bias - 1))) * mantissa_shift
        return ((-1) ** sign) * (2 ** (exponent - self.exp_bias)) * fmantissa

    def enumerate_all(self):
        nums = []
        hex_nums = []
        for s in [Sign.positive, Sign.negative]:
            for i in range((1 << self.we)):
                for j in range((1 << self.wm)):
                    hex_nums.append(hex((s << self.we | i)
                                        << self.wm | j))
                    nums.append(self.construct_fp8(j, i, s))
        return zip(hex_nums, nums)


if __name__ == "__main__":
    fp8e4m3fnuz = Float8(3, 4, True, False, Rounding.standard)
    fp8e5m2fnuz = Float8(2, 5, True, False, Rounding.standard)
    all_fp8e5m2fnuz = fp8e5m2fnuz.enumerate_all()
    all_fp8e4m3fnuz = fp8e4m3fnuz.enumerate_all()
    print(list(all_fp8e5m2fnuz))
