"""Write a number out by hand in fp32, bf16 and fp8 E4M3, showing the bits.

Nothing here calls a "to_fp8" library. fp32 and bf16 come straight from the IEEE
layout via numpy; fp8 E4M3 is done by hand (round-to-nearest-even on 3 mantissa
bits) so you can see every step. The default number is 0.1 -- the classic
"looks exact, isn't" value.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np


@dataclass
class FloatBreakdown:
    fmt: str
    sign_bit: str
    exp_bits: str
    mant_bits: str
    bias: int
    unbiased_exp: int
    mantissa_fraction: float  # the 1.xxxx significand
    stored_value: float       # what the format actually represents
    target: float             # the number we asked for

    @property
    def bit_string(self) -> str:
        return f"{self.sign_bit} {self.exp_bits} {self.mant_bits}"

    @property
    def abs_error(self) -> float:
        return abs(self.stored_value - self.target)

    @property
    def rel_error(self) -> float:
        return self.abs_error / abs(self.target) if self.target else 0.0

    def render(self) -> str:
        return (
            f"{self.fmt:>9} | {self.bit_string:<40} | "
            f"(-1)^{self.sign_bit} x {self.mantissa_fraction:.10f} x 2^{self.unbiased_exp:<4} "
            f"= {self.stored_value:.12f}  (rel err {self.rel_error:.2e})"
        )


def fp32_breakdown(x: float) -> FloatBreakdown:
    bits = struct.unpack(">I", struct.pack(">f", x))[0]
    b = f"{bits:032b}"
    sign, exp, mant = b[0], b[1:9], b[9:]
    exp_val = int(exp, 2)
    unbiased = exp_val - 127
    frac = 1.0 + int(mant, 2) / (1 << 23)
    stored = struct.unpack(">f", struct.pack(">f", x))[0]
    return FloatBreakdown("fp32", sign, exp, mant, 127, unbiased, frac, float(stored), x)


def bf16_breakdown(x: float) -> FloatBreakdown:
    # bf16 = fp32 truncated (round-to-nearest-even) to the top 16 bits.
    f32 = np.float32(x)
    u32 = f32.view(np.uint32)
    # round to nearest even on the 16 discarded bits
    rounding_bias = np.uint32(0x7FFF) + ((u32 >> np.uint32(16)) & np.uint32(1))
    u16 = np.uint16((u32 + rounding_bias) >> np.uint32(16))
    b = f"{int(u16):016b}"
    sign, exp, mant = b[0], b[1:9], b[9:]
    exp_val = int(exp, 2)
    unbiased = exp_val - 127
    frac = 1.0 + int(mant, 2) / (1 << 7)
    stored = np.uint32(int(u16) << 16).view(np.float32)
    return FloatBreakdown("bf16", sign, exp, mant, 127, unbiased, frac, float(stored), x)


def fp8_e4m3_breakdown(x: float) -> FloatBreakdown:
    """fp8 E4M3: 1 sign, 4 exponent (bias 7), 3 mantissa. Done by hand.

    E4M3 is *not* IEEE: there is no inf, and the largest exponent field is used
    for finite numbers (max normal 448). We only need the normal-range path for
    0.1, but subnormals are handled for completeness.
    """
    sign_bit = "1" if x < 0 else "0"
    ax = abs(float(x))
    bias = 7

    if ax == 0.0:
        return FloatBreakdown("fp8_e4m3", sign_bit, "0000", "000", bias, -bias, 0.0, 0.0, x)

    # unbiased exponent e such that 1.0 <= ax / 2^e < 2.0
    e = int(np.floor(np.log2(ax)))
    exp_field = e + bias

    if exp_field <= 0:
        # subnormal: exponent field 0, no implicit 1
        step = 2.0 ** (1 - bias - 3)  # smallest subnormal
        q = round(ax / step)
        q = min(q, 7)
        mant_int = q
        exp_field = 0
        frac = mant_int / 8.0
        stored = mant_int * step
        unbiased = 1 - bias
    else:
        significand = ax / (2.0 ** e)            # in [1, 2)
        mant_scaled = (significand - 1.0) * 8.0  # in [0, 8)
        mant_int = int(np.floor(mant_scaled))
        remainder = mant_scaled - mant_int
        # round to nearest, ties to even
        if remainder > 0.5 or (remainder == 0.5 and mant_int % 2 == 1):
            mant_int += 1
        if mant_int == 8:  # carry into the exponent
            mant_int = 0
            exp_field += 1
            e += 1
        # E4M3 caps at exp_field 15 with mantissa 110 (value 448); clamp
        if exp_field >= 16:
            exp_field, mant_int = 15, 6
        frac = 1.0 + mant_int / 8.0
        stored = frac * (2.0 ** (exp_field - bias))
        unbiased = exp_field - bias

    return FloatBreakdown(
        "fp8_e4m3",
        sign_bit,
        f"{exp_field:04b}",
        f"{mant_int:03b}",
        bias,
        unbiased,
        frac,
        float(stored),
        x,
    )


def all_breakdowns(x: float = 0.1) -> list[FloatBreakdown]:
    return [fp32_breakdown(x), bf16_breakdown(x), fp8_e4m3_breakdown(x)]


def report(x: float = 0.1) -> str:
    lines = [f"the number {x!r}, written out by hand:\n"]
    for bd in all_breakdowns(x):
        lines.append(bd.render())
    return "\n".join(lines)


if __name__ == "__main__":
    print(report(0.1))
