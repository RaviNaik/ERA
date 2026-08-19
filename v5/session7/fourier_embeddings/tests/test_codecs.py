"""Sanity tests that tie the code directly back to the research note's
theorems, so a change to the codec implementation that breaks the math it is
supposed to implement fails loudly and immediately (not just "the LM run
looked a bit worse").
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from fourier_embeddings.model.codecs import (
    ByteGridCodec, DenseEmbedding, HRRBindingEmbedding, KroneckerEmbedding,
    FourierKroneckerEmbedding, build_byte_tables, sinusoidal_position_table,
)


def _pre_projection_codes(codec: ByteGridCodec, words: list[str]) -> torch.Tensor:
    byte_lists = [w.encode("utf-8") for w in words]
    # byte_capacity, not pos_dim -- the two differ for pos_factor="fourier"
    # since the fix for the truncation-wall gap (research note Sec. 6.1):
    # Kronecker's byte_capacity always equals pos_dim, but Fourier's
    # defaults to the longest real token, so using pos_dim here would build
    # a differently-shaped table than the codec's own buffers for Fourier.
    byte_ids, valid_mask, length = build_byte_tables(byte_lists, codec.byte_capacity)
    return codec._grid_codes(byte_ids, valid_mask, length)


def test_kronecker_theorem1_equal_length_disagreement():
    """Theorem 1: cos(kappa(a), kappa(b)) = (L-k)/L for equal-length strings
    agreeing at L-k positions, *before* the final z-normalization step.
    `mistake` vs `mistkae`: L=7, one transposition (2 positions differ)
    -> cos = 5/7 (the exact number the research note quotes from the paper).
    """
    words = ["mistake", "mistkae"]
    byte_ids, valid_mask, length = build_byte_tables([w.encode() for w in words], 8)
    pos_vecs = torch.eye(8)
    weight = (1.0 / length.float().sqrt()).unsqueeze(-1) * valid_mask.float()
    contrib = weight.unsqueeze(-1) * pos_vecs.unsqueeze(0)
    grid = torch.zeros(2, 256, 8)
    index = byte_ids.unsqueeze(-1).expand(2, 8, 8)
    grid.scatter_add_(1, index, contrib)
    raw = grid.reshape(2, -1)
    raw_cos = F.cosine_similarity(raw[0:1], raw[1:2]).item()

    L, k = 7, 2
    expected = (L - k) / L
    assert math.isclose(raw_cos, expected, abs_tol=1e-5), f"{raw_cos} != {expected}"

    # The full codec (with z-normalization) should still rank this pair as
    # highly similar -- z-normalization is a monotonic-in-spirit rescaling
    # for this sparse, equal-magnitude-nonzero-entries construction.
    codec = ByteGridCodec(len(words), 8, [w.encode() for w in words], pos_dim=8, pos_factor="onehot")
    codes = _pre_projection_codes(codec, words)
    cos = F.cosine_similarity(codes[0:1], codes[1:2]).item()
    assert cos > 0.5


def test_kronecker_derangement_exact_orthogonality_pre_normalization():
    """Corollary (Sec. 2): a positional rearrangement with *no fixed point*
    (a derangement -- every byte lands at a position it didn't start at)
    shares no (value, position) cell, so the pre-z-normalization inner
    product is exactly 0.

    NOTE: the research note's own worked example for this corollary uses
    "cat" vs. "tac" (Sec. 2, Sec. 1.3) and claims cos=0. That specific pair
    is *not* actually a derangement: reversing an odd-length string fixes
    its middle character, so "cat"[1]='a' and "tac"[1]='a' share position 1.
    Theorem 1's own formula confirms this: they agree at 1 of 3 positions,
    so cos = (3-1)/3 = 1/3, not 0 -- and Sec. 4.3's Fourier worked example
    for the *same* pair independently arrives at the same conclusion,
    explicitly treating "a@1 matches a@1 at the same position" as a direct
    (non-cross) term. This is confirmed empirically by
    `fe-order-sensitivity`'s output for the ("cat","tac") pair (see
    README/report). The corollary itself is correct for true derangements,
    which is what this test verifies with an unambiguous example instead.
    """
    words = ["abcde", "bcdea"]  # cyclic rotation by 1: a true derangement, no fixed points
    byte_ids, valid_mask, length = build_byte_tables([w.encode() for w in words], 5)
    pos_vecs = torch.eye(5)
    weight = (1.0 / length.float().sqrt()).unsqueeze(-1) * valid_mask.float()
    contrib = weight.unsqueeze(-1) * pos_vecs.unsqueeze(0)
    grid = torch.zeros(2, 256, 5)
    index = byte_ids.unsqueeze(-1).expand(2, 5, 5)
    grid.scatter_add_(1, index, contrib)
    raw = grid.reshape(2, -1)
    dot = (raw[0] * raw[1]).sum().item()
    assert math.isclose(dot, 0.0, abs_tol=1e-6)


def test_fourier_gives_nonzero_anagram_similarity():
    """Design A (Sec. 4.3): the same 'cat' vs 'tac' pair should get a
    *nonzero* similarity once the position factor is Fourier instead of
    one-hot -- the concrete, numeric face of the order-sensitivity trade.
    """
    words = ["cat", "tac"]
    codec = ByteGridCodec(len(words), 8, [w.encode() for w in words], pos_dim=4,
                           pos_factor="fourier", pos_factor_dim=4)
    codes = _pre_projection_codes(codec, words)
    cos = F.cosine_similarity(codes[0:1], codes[1:2]).item()
    assert cos > 1e-4, f"expected nonzero similarity, got {cos}"


def test_position_kernel_maximal_at_zero_offset():
    """Proposition 1 (Sec. 3.2): K(delta) = sum cos(omega_i delta) is maximal
    at delta=0 and strictly smaller elsewhere (for the standard schedule).
    """
    phi = sinusoidal_position_table(max_len=32, d_p=8, schedule="standard")
    K = phi @ phi.T  # [32, 32] kernel matrix, K[i,j] = phi(i).phi(j)
    diag = K.diagonal()
    assert torch.allclose(diag, diag[0] * torch.ones_like(diag), atol=1e-4)
    off_diag_max = (K - torch.diag(diag)).max().item()
    assert off_diag_max < diag[0].item() - 1e-4


def test_truncation_wall_removed_for_fourier():
    """Sec. 6.1: a one-hot Kronecker codec must silently crop bytes past
    pos_dim (two tokens agreeing on the first pos_dim bytes collide exactly);
    the Fourier codec must not -- phi(p) is defined for every p, so extending
    a shared prefix with *different* tails must change the code.

    This is a real end-to-end check, not just a check that phi(p) *can* be
    evaluated past pos_dim in isolation. Earlier, ByteGridCodec built its
    byte buffers at a fixed width of pos_dim for *both* codecs regardless of
    pos_factor, so the Fourier arm silently cropped exactly like Kronecker
    does, contradicting this exact claim (caught in strict review, see the
    project README's revision history). The fix: byte_capacity (the actual
    buffer width) now defaults to the longest real token for pos_factor=
    "fourier" specifically, decoupled from pos_dim (which continues to only
    shape the frequency schedule). Kronecker's byte_capacity always equals
    pos_dim -- its hard cliff is unchanged and re-verified below too.
    """
    pos_dim = 8
    prefix = "abcdefgh"  # exactly pos_dim bytes
    words = [prefix + "X", prefix + "Y"]  # 9 bytes each, differ only past pos_dim

    kron = ByteGridCodec(len(words), 8, [w.encode() for w in words], pos_dim=pos_dim, pos_factor="onehot")
    assert kron.byte_capacity == pos_dim, "Kronecker's byte_capacity must stay tied to pos_dim"
    kron_codes = _pre_projection_codes(kron, words)
    assert torch.allclose(kron_codes[0], kron_codes[1], atol=1e-5), \
        "Kronecker codec should still collide exactly on a shared pos_dim-byte prefix"

    fourier = ByteGridCodec(len(words), 8, [w.encode() for w in words], pos_dim=pos_dim,
                             pos_factor="fourier", pos_factor_dim=8)
    assert fourier.byte_capacity >= 9, \
        f"Fourier byte_capacity should auto-size to the longest token (9 bytes), got {fourier.byte_capacity}"
    fourier_codes = _pre_projection_codes(fourier, words)
    assert not torch.allclose(fourier_codes[0], fourier_codes[1], atol=1e-4), \
        "Fourier codec must NOT collide on a prefix shared only up to pos_dim -- it should see the " \
        "differing 9th byte and produce different codes, which is the actual claim Sec. 6.1 makes"

    # phi(p) itself remains constructible for arbitrarily large p, independent
    # of any particular vocabulary's longest token.
    phi = sinusoidal_position_table(max_len=10_000, d_p=8, schedule="standard")
    assert phi.shape[0] == 10_000, "phi(p) must be constructible for arbitrarily large p"


def test_fourier_byte_capacity_explicit_override():
    """--byte-capacity lets a caller pin the Fourier buffer width instead of
    auto-detecting it (e.g. to force a *smaller* width than the vocabulary's
    longest token, deliberately reproducing the cropping this fix removes by
    default -- useful for an ablation, not just an escape hatch).
    """
    words = ["abcdefghij"]  # 10 bytes
    fourier = ByteGridCodec(len(words), 8, [w.encode() for w in words], pos_dim=4,
                             pos_factor="fourier", pos_factor_dim=4, byte_capacity=6)
    assert fourier.byte_capacity == 6
    assert fourier.byte_ids.shape[1] == 6


def test_onehot_rejects_mismatched_byte_capacity():
    """Kronecker's one-hot position factor has exactly pos_dim rows by
    construction -- passing a byte_capacity that disagrees with pos_dim must
    raise, not silently index out of bounds or silently ignore the argument.
    """
    words = ["hello"]
    try:
        ByteGridCodec(len(words), 8, [w.encode() for w in words], pos_dim=8,
                      pos_factor="onehot", byte_capacity=16)
        assert False, "expected a ValueError for onehot byte_capacity != pos_dim"
    except ValueError:
        pass


def test_cached_and_dynamic_modes_agree():
    words = ["training", "trainer", "the", "kronecker", "fourier"]
    id_to_bytes = [w.encode() for w in words]
    dyn = KroneckerEmbedding(len(words), 16, id_to_bytes, pos_dim=8, mode="dynamic")
    cached = KroneckerEmbedding(len(words), 16, id_to_bytes, pos_dim=8, mode="cached")
    cached.projection.weight.data = dyn.projection.weight.data.clone()

    ids = torch.arange(len(words)).unsqueeze(0)
    out_dyn = dyn(ids)
    out_cached = cached(ids)
    assert torch.allclose(out_dyn, out_cached, atol=1e-4)


def test_no_vocab_size_in_codec_param_count():
    """The headline Kronecker/Fourier property: projection param count must
    not depend on vocab_size (research note Sec. 1.1, "Job 1").
    """
    words_small = [f"tok{i}" for i in range(50)]
    words_large = [f"tok{i}" for i in range(5000)]
    small = KroneckerEmbedding(len(words_small), 32, [w.encode() for w in words_small], pos_dim=8)
    large = KroneckerEmbedding(len(words_large), 32, [w.encode() for w in words_large], pos_dim=8)
    small_params = sum(p.numel() for p in small.projection.parameters())
    large_params = sum(p.numel() for p in large.projection.parameters())
    assert small_params == large_params


def test_hrr_codec_runs_and_shapes_are_correct():
    words = ["cat", "tac", "dog"]
    codec = HRRBindingEmbedding(len(words), 16, [w.encode() for w in words], D=64, pos_dim=4)
    ids = torch.arange(len(words)).unsqueeze(0)
    out = codec(ids)
    assert out.shape == (1, len(words), 16)


def test_dense_embedding_baseline_shape():
    codec = DenseEmbedding(vocab_size=100, d_model=32)
    ids = torch.randint(0, 100, (2, 5))
    out = codec(ids)
    assert out.shape == (2, 5, 32)


def test_fourier_kronecker_end_to_end_shape():
    words = ["hello", "world", "transformer"]
    codec = FourierKroneckerEmbedding(len(words), 24, [w.encode() for w in words],
                                       pos_dim=8, fourier_dim=8)
    ids = torch.arange(len(words)).unsqueeze(0)
    out = codec(ids)
    assert out.shape == (1, len(words), 24)
