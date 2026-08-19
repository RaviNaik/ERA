"""
Byte-level token embedding codecs.

This module implements the embedding codecs compared by this project's
experiments, following `fourier_embeddings_research.md`:

  - `DenseEmbedding`             — the "Normal Embedding" control arm
                                   (plain `nn.Embedding`, one learned row/token).
  - `KroneckerEmbedding`         — the shipped byte-Kronecker codec (class notes
                                   S7 / Shravan 2026): a one-hot byte-value
                                   factor Kronecker-producted with a one-hot
                                   byte-*position* factor. The position factor
                                   is a delta kernel: pos[i].pos[j] = delta_ij.
  - `FourierKroneckerEmbedding`  — "Design A" from the research note: the same
                                   recipe, with the one-hot position factor
                                   replaced by a fixed multi-scale sinusoidal
                                   feature map phi(p) (a smooth, shift-invariant
                                   kernel instead of a delta kernel).
  - `HRRBindingEmbedding`        — "Design B": holographic reduced-representation
                                   binding (circular convolution of phase-only
                                   spectra) instead of a Kronecker product at all.

`KroneckerEmbedding` and `FourierKroneckerEmbedding` share one implementation
(`ByteGridCodec`): the only thing that differs between them is the table of
per-position feature vectors used as the "position factor". This mirrors
research-note Sec. 5's claim that Kronecker is the zero-bandwidth degenerate
case of the Fourier construction, not a competing mechanism.

Every codec exposes the same contract as `nn.Embedding`:
    input_ids: LongTensor[B, T]  ->  FloatTensor[B, T, d_model]
so any of them can be swapped into the GPT model unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


# --------------------------------------------------------------------------
# Byte-table construction shared by every byte-level codec
# --------------------------------------------------------------------------

def build_byte_tables(
    id_to_bytes: list[bytes],
    pos_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Turn a vocabulary's token->bytes map into fixed lookup buffers.

    Returns:
        byte_ids:   LongTensor[V, pos_dim]  byte value (0-255) at each grid
                    column; 0 in padding/truncated columns (masked out below).
        valid_mask: BoolTensor[V, pos_dim]  True where a real byte lives.
        length:     LongTensor[V]           min(len(token_bytes), pos_dim),
                    i.e. L in the codec formula. Clamped to >= 1 so empty
                    tokens never cause a division by zero.
    """
    V = len(id_to_bytes)
    byte_ids = torch.zeros(V, pos_dim, dtype=torch.long)
    valid_mask = torch.zeros(V, pos_dim, dtype=torch.bool)
    length = torch.ones(V, dtype=torch.long)

    for v, b in enumerate(id_to_bytes):
        L = min(len(b), pos_dim)
        length[v] = max(L, 1)
        if L > 0:
            byte_ids[v, :L] = torch.tensor(list(b[:L]), dtype=torch.long)
            valid_mask[v, :L] = True

    return byte_ids, valid_mask, length


def sinusoidal_position_table(
    max_len: int,
    d_p: int,
    schedule: Literal["standard", "narrow"] = "standard",
    base: float = 10000.0,
) -> torch.Tensor:
    """phi(p) for p = 0..max_len-1, shape [max_len, d_p].

    `d_p` must be even (m = d_p/2 frequency bands, sin/cos pair each).

    schedule="standard": the log-linear multi-octave schedule from research
        note Sec. 3.2 / Sec. 8.1's mitigation:  omega_i = 1 / base^(2i/d_p).
    schedule="narrow": a deliberately narrow-band schedule (all frequencies
        clustered near the low end) used to empirically reproduce the Sec. 8.1
        failure mode ("too narrow-band, nearby positions barely distinguishable")
        as one arm of the controlled comparison (research note Sec. 10, item 1c).
    """
    assert d_p % 2 == 0, "d_p (Fourier position width) must be even"
    m = d_p // 2
    p = torch.arange(max_len, dtype=torch.float64).unsqueeze(1)  # [max_len, 1]
    i = torch.arange(m, dtype=torch.float64).unsqueeze(0)  # [1, m]

    if schedule == "standard":
        omega = 1.0 / (base ** (2.0 * i / d_p))  # log-linear octave spread, spans ~[1/base, 1]
    elif schedule == "narrow":
        # Deliberately narrow-band: every frequency crowded within a factor of
        # ~2 of a single low value, instead of spanning several octaves. This
        # is the exact failure mode research-note Sec. 8.1 predicts ("only low
        # frequencies present, so nearby positions are barely distinguishable
        # -- a soft version of the collision problem this design was meant to
        # fix"), reproduced here so the controlled comparison can measure it
        # rather than merely assert it.
        omega_low = 1.0 / base  # same low end as the standard schedule's slowest band
        omega = omega_low * (1.0 + i / max(m - 1, 1))  # spans [omega_low, 2*omega_low]
    else:
        raise ValueError(f"unknown schedule: {schedule}")

    angles = p * omega  # [max_len, m]
    phi = torch.zeros(max_len, d_p, dtype=torch.float64)
    phi[:, 0::2] = torch.sin(angles)
    phi[:, 1::2] = torch.cos(angles)
    return phi.float()


# --------------------------------------------------------------------------
# Design principle (research note Sec. 5): one generic "value factor (x)
# position factor, summed over bytes, projected once" recipe. Kronecker and
# Fourier differ only in which table is plugged in as the position factor.
# --------------------------------------------------------------------------

class ByteGridCodec(nn.Module):
    """Shared implementation for the one-hot (Kronecker) and Fourier position
    factors. See module docstring.

    kappa(b) = z_normalize( (1/sqrt(L)) * sum_p  onehot(byte_p) (x) pos_factor(p) )
    output   = Linear(D, d_model, bias=False)(kappa(b))
        where D = char_dim * pos_factor_dim
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        id_to_bytes: list[bytes],
        char_dim: int = 256,
        pos_dim: int = 32,
        pos_factor_dim: int | None = None,
        pos_factor: Literal["onehot", "fourier"] = "onehot",
        fourier_schedule: Literal["standard", "narrow"] = "standard",
        mode: Literal["dynamic", "cached"] = "dynamic",
        byte_capacity: int | None = None,
    ):
        """`byte_capacity` is the actual buffer width used for `build_byte_tables`
        and, for `pos_factor="fourier"`, the length phi(p) is tabulated out to.
        This is the fix for research-note Sec. 6.1's "no truncation wall"
        claim: `pos_dim` alone used to double as *both* the Fourier position
        factor's bandwidth *and* the byte-buffer crop width, so the Fourier
        codec silently cropped tokens at `pos_dim` exactly like Kronecker
        does, contradicting Theorem 2 (phi(p) is defined for every p) even
        though the math was never wrong -- only this module's buffer layout
        was. The two are now decoupled:
          - `pos_factor="onehot"` (Kronecker): `byte_capacity` MUST equal
            `pos_dim` -- one-hot's position table has exactly `pos_dim` rows
            by construction, so `pos_dim` genuinely *is* the coverage budget
            here, unlike for Fourier. This preserves Kronecker's hard cliff
            exactly as before; passing anything else raises.
          - `pos_factor="fourier"`: `byte_capacity` defaults to the longest
            token actually present in `id_to_bytes` (computed once, here),
            so no real vocabulary token is ever cropped, full stop -- not
            "less likely to be cropped." `pos_dim` continues to shape only
            the frequency schedule via `pos_factor_dim` (`fourier_dim`),
            exactly as documented; it no longer bounds the byte buffer.
            Pass an explicit `byte_capacity` to cap this at a fixed value
            instead (e.g. if one pathological token would otherwise force a
            very wide buffer for the whole vocabulary).
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.char_dim = char_dim
        self.pos_dim = pos_dim
        self.pos_factor_kind = pos_factor
        self.mode = mode

        if pos_factor == "onehot":
            if byte_capacity is not None and byte_capacity != pos_dim:
                raise ValueError(
                    f"onehot (Kronecker) position factor has exactly pos_dim={pos_dim} rows "
                    f"by construction -- byte_capacity={byte_capacity} would index out of "
                    f"bounds. pos_dim IS Kronecker's coverage budget (research note Sec. 2); "
                    f"only the fourier position factor can decouple byte_capacity from pos_dim."
                )
            cap = pos_dim
            pos_factor_dim = pos_dim
            table = torch.eye(pos_dim)
        elif pos_factor == "fourier":
            assert pos_factor_dim is not None, "pos_factor_dim required for fourier"
            if byte_capacity is not None:
                cap = byte_capacity
            else:
                longest = max((len(b) for b in id_to_bytes), default=0)
                cap = max(longest, pos_dim, 1)  # never shrink below pos_dim or 1
            table = sinusoidal_position_table(cap, pos_factor_dim, schedule=fourier_schedule)
        else:
            raise ValueError(pos_factor)
        self.byte_capacity = cap
        self.pos_factor_dim = pos_factor_dim
        self.register_buffer("pos_factor_table", table, persistent=False)  # [byte_capacity, pos_factor_dim]

        byte_ids, valid_mask, length = build_byte_tables(id_to_bytes, cap)
        self.register_buffer("byte_ids", byte_ids, persistent=False)
        self.register_buffer("valid_mask", valid_mask, persistent=False)
        self.register_buffer("length", length, persistent=False)

        self.D = char_dim * pos_factor_dim
        self.projection = nn.Linear(self.D, d_model, bias=False)
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.02)

        self._cached_codes: torch.Tensor | None = None  # [V, D], built lazily

    # -- core math, shared by dynamic and cached paths --------------------

    def _grid_codes(self, byte_ids: torch.Tensor, valid_mask: torch.Tensor,
                     length: torch.Tensor) -> torch.Tensor:
        """Build z-normalized, flattened codes for an arbitrary batch of
        (byte_ids, valid_mask, length) with shape [..., pos_dim] / [...].
        Returns FloatTensor[..., D].
        """
        *lead, P = byte_ids.shape
        weight = (1.0 / length.float().sqrt()).unsqueeze(-1)  # [..., 1]
        weight = weight * valid_mask.float()  # zero out padded columns

        pos_vecs = self.pos_factor_table[torch.arange(P, device=byte_ids.device)]
        pos_vecs = pos_vecs.to(byte_ids.device)  # [pos_dim, pos_factor_dim]
        pos_vecs = pos_vecs.expand(*lead, P, self.pos_factor_dim)

        contribution = weight.unsqueeze(-1) * pos_vecs  # [..., pos_dim, pos_factor_dim]

        grid = byte_ids.new_zeros(*lead, self.char_dim, self.pos_factor_dim, dtype=torch.float32)
        index = byte_ids.unsqueeze(-1).expand(*lead, P, self.pos_factor_dim)
        grid.scatter_add_(-2, index, contribution)

        code = grid.reshape(*lead, self.D)

        # z-normalize each token's code (research note Sec. 2 / class notes Sec. 7)
        mean = code.mean(dim=-1, keepdim=True)
        std = code.std(dim=-1, unbiased=False, keepdim=True)
        code = (code - mean) / (std + EPS)
        return code

    def build_cache(self):
        """Precompute the per-vocab-id code table (class notes' `mode="cached"`).
        Deterministic given (tokenizer, char_dim, pos_dim, position factor), so
        it is safe to build once and reuse for the module's lifetime.
        """
        with torch.no_grad():
            codes = self._grid_codes(self.byte_ids, self.valid_mask, self.length)  # [V, D]
        self._cached_codes = codes
        return codes

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.mode == "cached":
            if self._cached_codes is None or self._cached_codes.device != input_ids.device:
                self.build_cache()
                self._cached_codes = self._cached_codes.to(input_ids.device)
            code = F.embedding(input_ids, self._cached_codes)
        else:
            byte_ids = self.byte_ids[input_ids]
            valid_mask = self.valid_mask[input_ids]
            length = self.length[input_ids]
            code = self._grid_codes(byte_ids, valid_mask, length)
        return self.projection(code)

    def extra_repr(self) -> str:
        return (f"vocab_size={self.vocab_size}, char_dim={self.char_dim}, pos_dim={self.pos_dim}, "
                f"byte_capacity={self.byte_capacity}, pos_factor={self.pos_factor_kind}, "
                f"pos_factor_dim={self.pos_factor_dim}, D={self.D}, d_model={self.d_model}, "
                f"mode={self.mode}")


def KroneckerEmbedding(
    vocab_size: int,
    d_model: int,
    id_to_bytes: list[bytes],
    char_dim: int = 256,
    pos_dim: int = 32,
    mode: Literal["dynamic", "cached"] = "dynamic",
) -> ByteGridCodec:
    """The shipped one-hot Kronecker codec (class notes Sec. 7 / Widget 8)."""
    return ByteGridCodec(
        vocab_size=vocab_size, d_model=d_model, id_to_bytes=id_to_bytes,
        char_dim=char_dim, pos_dim=pos_dim, pos_factor="onehot", mode=mode,
    )


def FourierKroneckerEmbedding(
    vocab_size: int,
    d_model: int,
    id_to_bytes: list[bytes],
    char_dim: int = 256,
    pos_dim: int = 32,
    fourier_dim: int = 32,
    schedule: Literal["standard", "narrow"] = "standard",
    mode: Literal["dynamic", "cached"] = "dynamic",
    byte_capacity: int | None = None,
) -> ByteGridCodec:
    """Design A: Fourier-position Kronecker codec (research note Sec. 4.2).

    `pos_dim` here is the *bandwidth budget* used to build phi's frequency
    schedule (kept for parity with Kronecker's naming) — unlike one-hot
    Kronecker, phi(p) is defined for every p, so there is no truncation wall;
    `pos_dim` only shapes the schedule, it does not crop tokens. This is now
    actually true of the byte buffer too, not just of phi(p) in isolation:
    `byte_capacity` (see `ByteGridCodec.__init__`) defaults to the longest
    token in `id_to_bytes`, so no real vocabulary token is cropped. Pass an
    explicit `byte_capacity` to cap this at a fixed width instead.
    """
    return ByteGridCodec(
        vocab_size=vocab_size, d_model=d_model, id_to_bytes=id_to_bytes,
        char_dim=char_dim, pos_dim=pos_dim, pos_factor="fourier",
        pos_factor_dim=fourier_dim, fourier_schedule=schedule, mode=mode,
        byte_capacity=byte_capacity,
    )


# --------------------------------------------------------------------------
# Design B: holographic reduced representation (HRR) binding
# --------------------------------------------------------------------------

def _unit_phase_spectrum(n: int, D: int, generator: torch.Generator) -> torch.Tensor:
    """n random real vectors of dim D whose FFT has unit magnitude at every
    frequency ("phase-only spectra"), required for exact invertibility of
    circular-convolution binding (research note Sec. 3.4).
    """
    phases = torch.rand(n, D // 2 + 1, generator=generator) * 2 * math.pi
    mags = torch.ones_like(phases)
    spectrum = torch.polar(mags, phases)  # complex64, unit magnitude
    # Force DC and Nyquist (if D even) to be real, so the inverse FFT is real.
    spectrum[:, 0] = 1.0
    if D % 2 == 0:
        spectrum[:, -1] = 1.0
    vecs = torch.fft.irfft(spectrum, n=D)
    return vecs


class HRRBindingEmbedding(nn.Module):
    """Design B (research note Sec. 4.4): kappa(b) = sum_p c_wave[byte_p] (*) pos_wave[p].

    `c_wave` (256 byte-value vectors) and `pos_wave` (byte_capacity position
    vectors) are fixed, seeded, phase-only-spectrum random vectors in R^D.
    `D` is a free width, fully decoupled from char_dim/pos_dim/vocab (no grid
    to flatten at all). Only the downstream projection is learned.

    `pos_dim`/`byte_capacity` split the same way as `ByteGridCodec` (see its
    docstring): HRR's per-position vectors are independent random draws with
    no mathematical constraint tying their *count* to `pos_dim` (unlike
    Kronecker's one-hot position factor), so extending `pos_wave` to cover
    every real token's length costs nothing but a few more random rows and
    fixes the same "no truncation wall" gap Sec. 6.1 claims for Design B too.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        id_to_bytes: list[bytes],
        D: int = 1024,
        pos_dim: int = 32,
        char_dim: int = 256,
        seed: int = 0,
        byte_capacity: int | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.D = D
        self.pos_dim = pos_dim
        self.char_dim = char_dim

        if byte_capacity is not None:
            cap = byte_capacity
        else:
            longest = max((len(b) for b in id_to_bytes), default=0)
            cap = max(longest, pos_dim, 1)
        self.byte_capacity = cap

        byte_ids, valid_mask, length = build_byte_tables(id_to_bytes, cap)
        self.register_buffer("byte_ids", byte_ids, persistent=False)
        self.register_buffer("valid_mask", valid_mask, persistent=False)
        self.register_buffer("length", length, persistent=False)

        gen = torch.Generator().manual_seed(seed)
        c_wave = _unit_phase_spectrum(char_dim, D, gen)
        pos_wave = _unit_phase_spectrum(cap, D, gen)
        self.register_buffer("c_wave", c_wave, persistent=False)  # [256, D]
        self.register_buffer("pos_wave", pos_wave, persistent=False)  # [byte_capacity, D]

        self.projection = nn.Linear(D, d_model, bias=False)
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.02)

    def bind_sum(self, byte_ids: torch.Tensor, valid_mask: torch.Tensor,
                 length: torch.Tensor) -> torch.Tensor:
        """sum_p c_wave[byte_p] (*) pos_wave[p], via the FFT convolution
        theorem (elementwise product of spectra), for an arbitrary batch of
        (byte_ids, valid_mask, length) shaped [..., pos_dim] / [...].
        """
        *lead, P = byte_ids.shape
        weight = (1.0 / length.float().sqrt()).unsqueeze(-1) * valid_mask.float()  # [..., pos_dim]

        c_spec = torch.fft.rfft(self.c_wave, dim=-1)  # [256, D//2+1]
        p_spec = torch.fft.rfft(self.pos_wave, dim=-1)  # [pos_dim, D//2+1]

        c_sel = c_spec[byte_ids]  # [..., pos_dim, D//2+1]
        p_sel = p_spec[torch.arange(P, device=byte_ids.device)]
        p_sel = p_sel.expand(*lead, P, p_spec.shape[-1])

        bound_spec = c_sel * p_sel  # elementwise product == circular convolution
        bound_spec = bound_spec * weight.unsqueeze(-1)
        summed_spec = bound_spec.sum(dim=-2)  # sum over the pos_dim axis -> [..., D//2+1]
        code = torch.fft.irfft(summed_spec, n=self.D)  # [..., D]
        return code

    def unbind(self, code: torch.Tensor, position: int) -> torch.Tensor:
        """Approximate recovery of the byte value's wave at `position` from a
        bound-and-summed code, via correlation with pos_wave[position]^{-1}
        (complex conjugate, since pos_wave has unit-magnitude spectrum).
        Used only by the crosstalk analysis script, not by the forward pass.
        """
        code_spec = torch.fft.rfft(code, dim=-1)
        p_spec = torch.fft.rfft(self.pos_wave[position], dim=-1)
        recovered_spec = code_spec * p_spec.conj()
        return torch.fft.irfft(recovered_spec, n=self.D)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        byte_ids = self.byte_ids[input_ids]
        valid_mask = self.valid_mask[input_ids]
        length = self.length[input_ids]
        code = self.bind_sum(byte_ids, valid_mask, length)
        return self.projection(code)

    def extra_repr(self) -> str:
        return (f"vocab_size={self.vocab_size}, D={self.D}, pos_dim={self.pos_dim}, "
                f"byte_capacity={self.byte_capacity}, d_model={self.d_model}")


# --------------------------------------------------------------------------
# Control arm: the "Normal Embedding"
# --------------------------------------------------------------------------

class DenseEmbedding(nn.Module):
    """Plain `nn.Embedding` — one learned row per vocabulary token. This is
    the "Normal Embedding" baseline the user asked to compare against.
    """

    def __init__(self, vocab_size: int, d_model: int, **_ignored):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.emb(input_ids)


@dataclass
class CodecSpec:
    """Config knobs needed to build any of the codecs above from CLI args."""
    kind: Literal["dense", "kronecker", "fourier", "fourier_narrow", "hrr"]
    char_dim: int = 256
    pos_dim: int = 32
    fourier_dim: int = 32
    hrr_dim: int = 1024
    mode: Literal["dynamic", "cached"] = "dynamic"
    seed: int = 0
    byte_capacity: int | None = None  # fourier/hrr only; None = auto (longest real token)


def build_codec(spec: CodecSpec, vocab_size: int, d_model: int,
                 id_to_bytes: list[bytes]) -> nn.Module:
    if spec.kind == "dense":
        return DenseEmbedding(vocab_size, d_model)
    if spec.kind == "kronecker":
        return KroneckerEmbedding(vocab_size, d_model, id_to_bytes,
                                   char_dim=spec.char_dim, pos_dim=spec.pos_dim, mode=spec.mode)
    if spec.kind == "fourier":
        return FourierKroneckerEmbedding(vocab_size, d_model, id_to_bytes,
                                          char_dim=spec.char_dim, pos_dim=spec.pos_dim,
                                          fourier_dim=spec.fourier_dim, schedule="standard", mode=spec.mode,
                                          byte_capacity=spec.byte_capacity)
    if spec.kind == "fourier_narrow":
        return FourierKroneckerEmbedding(vocab_size, d_model, id_to_bytes,
                                          char_dim=spec.char_dim, pos_dim=spec.pos_dim,
                                          fourier_dim=spec.fourier_dim, schedule="narrow", mode=spec.mode,
                                          byte_capacity=spec.byte_capacity)
    if spec.kind == "hrr":
        return HRRBindingEmbedding(vocab_size, d_model, id_to_bytes,
                                    D=spec.hrr_dim, pos_dim=spec.pos_dim, char_dim=spec.char_dim,
                                    seed=spec.seed, byte_capacity=spec.byte_capacity)
    raise ValueError(f"unknown codec kind: {spec.kind}")


def codec_param_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
