"""The SHA3-256 implementation and some helper functions.

The functions are specified in NIST FIPS 202
(Link: <https://csrc.nist.gov/pubs/fips/202/final>)
"""

from bitstring import BitArray
from collections.abc import Callable
import numpy as np
import permutations as pmt

def sha3_256(message: bytes) -> str:
    """The SHA3-256 implementation.

    Parameters:
    -----------
    message: str
        The input to the hash function.

    Returns:
    digest: bytes
        a.k.a. hash value. The digest often serves as a condensed
        representation of the message.
    """
    m = BitArray(message) + BitArray(bin="01")
    return keccak(512, m, 256).tobytes().hex()


def keccak(c: int, m: BitArray, d: int) -> BitArray:
    """KECCAK is the family of sponge functions with the
    KECCAK-p[b, 2l +12] permutation.

    When restricted to the case `b = 1600`, the KECCAK family is denoted
    by KECCAK[c]; in this case `r` is determined by the choice of `c`.
    In particular,

    `KECCAK[c] (M, d) =
    SPONGE[KECCAK-p[1600, 24], pad10*1, 1600 – c] (M, d)`.

    Parameters:
    c: int
        The capacity of a sponge function.

    m: int
        The input message to a SHA-3 function.

    d: int
        The length of the digest of a hash function or the requested
        length of the output of an XOF, in bits.
    """
    return sponge(pad_10_asterisk_1, 1600 - c, m, d)


def keccak_p(s: BitArray, b: int=1600, rounds: int=24) -> BitArray:
    """The KECCAK-p[b, n_r] permutation consists of nr iterations of
    Rnd.

    Given a state array A and a round index ir, the round function Rnd
    is the transformation that results from applying the step mappings
    θ, ρ, π, χ, and ι, in that order, i.e.,:

    `Rnd(A, i_r) = ι(χ(π(ρ(θ(A)))), i_r)`.

    Parameters:
    -----------
    rounds: int
        Number of rounds n_r.

    s: BitArray
        String of length b.

    b: int, optional
        The default is 1600, which is the width of a KECCAK-p
        permutation in bits.

    Returns:
    --------
    s_prime: BitArray
        String of length b after the permutations.
    """
    state = _convert_to_state(s)

    start_index = 2 * pmt.L + 12 - rounds
    end_index = 2 * pmt.L + 12
    for i_r in range(start_index, end_index):
        state = _rnd(state, i_r)

    return _convert_to_bit_stream(state)


def _convert_to_state(s: BitArray) -> pmt.KeccakState:
    """Convert the bit array `s` to a KECCAK state for the permutation.

    For all triples (x, y, z) such that 0 <= x < 5, 0 <= y < 5,
    and 0 <= z < w,

    `A[x, y, z] = S[w(5y+x) + z]`.

    Parameters:
    s: BitArray
        The bit array of string.

    Returns:
    state: KeccakState:
        The state that is converted by `s`.
    """
    state = np.zeros(pmt.STATE_SHAPE, dtype=np.uint8)
    for x in range(pmt.STATE_X):
        for y in range(pmt.STATE_Y):
            for z in range(pmt.STATE_Z):
                state[x, y, z] = s[pmt.STATE_Z * (5 * y + x) + z]

    return state


def _convert_to_bit_stream(state: pmt.KeccakState) -> BitArray:
    """Convert the keccak state to the bit stream.

    For each pair of integers (i, j) such that 0 <= i < 5 and 0 <= j< 5,
    define the string Lane (i, j) by

    `Lane (i, j)= A[i, j, 0] || A[i, j, 1] || A[i, j, 2] || ... ||
    A[i, j, w-2] || A[i, j, w-1]`.

    For each integer j such that 0 <= j < 5, define the string Plane (j) by

    `Plane (j)= Lane (0, j) || Lane (1, j) || Lane (2, j) ||
    Lane (3, j) || Lane (4, j)`.

    Then

    `S= Plane (0) || Plane (1) || Plane (2) || Plane (3) || Plane (4)`.

    Parameters:
    state: KeccakState:
        The state that is converted by `s`.

    Returns:
    s: BitArray
        The bit array of string.
    """
    lane = np.zeros((pmt.STATE_X, pmt.STATE_Y), dtype=BitArray)
    for i in range(pmt.STATE_X):
        for j in range(pmt.STATE_Y):
            lane_i_j = BitArray()
            for k in range(pmt.STATE_Z):
                lane_i_j.append([state[i, j, k]])
            lane[i, j] = lane_i_j

    s = BitArray()
    for j in range(pmt.STATE_Y):
        for i in range(pmt.STATE_X):
            s.append(lane[i, j])

    return s


def _rnd(state: pmt.KeccakState, i_r: int) -> pmt.KeccakState:
    """The transformation that results from applying the step mappings
    θ, ρ, π, χ, and ι, in that order, i.e.,:

    `Rnd(A, ir) = ι(χ(π(ρ(θ(A)))), ir)`.

    Parameters:
    -----------
    state: KeccakState
        The current 5x5x64 keccak state cube.

    i_r: int
        The round index.

    Returns:
    --------
    updated_state: KeccakState
        The updated 5x5x64 state after the Rnd transformation.
    """
    return pmt.iota(pmt.chi(pmt.pi(pmt.rho(pmt.theta(state)))), i_r)


def sponge(pad: Callable[[int, int], BitArray], r: int, m: BitArray, d: int,
           b: int=1600) -> BitArray:
    """The sponge construction is a framework for specifying functions
    on binary data with arbitrary output length.

    The analogy to a sponge is that the function “absorbs” an arbitrary
    number of input bits into its state, after which an arbitrary number
    of output bits are “squeezed” out of its state.

    Parameters:
    -----------
    pad: Callable[[int, int], list[int]]
        Given a positive integer `x` and a non-negative integer `m`, the
        output `pad(x, m)` is a string with the property that
        `m + len(pad(x, m))` is a positive multiple of x.

    r: int
        The rate `r` is a positive integer that is strictly less
        than the width `b`.

    m: BitArray
        The input message to a SHA-3 function.

    d: int, nonnegative
        The length of the digest of a hash function or the requested
        length of the output of an XOF, in bits.

    b: int, optional
        The default is 1600, which is the width of a KECCAK-p
        permutation in bits.

    Returns:
    --------
    z: str
        String of length `d`.
    """
    p = m + pad(r, len(m))
    n = len(p) // r
    capacity = b - r
    s = BitArray(bin="0" * b)

    for i in range(n):
        p_i = p[i * r : (i + 1) * r]
        s = keccak_p(s ^ (p_i + BitArray(bin="0" * capacity)))

    z = BitArray()
    while True:
        z += s[:r]
        if d <= len(z):
            return z[:d]

        s = keccak_p(s)


def pad_10_asterisk_1(x: int, m: int) -> BitArray:
    """The `pad` function is used in `sponge()`.

    Let `j = (-m - 2) % x`, the return value is `11` with `j` times
    insertion of `0`.

    For example, if `j = 5`, the return value is `1000001`.

    Parameters:
    -----------
    x: int, positive
    m: int, nonnegative

    Returns:
    z: BitArray
        String Z such that m + len(Z) is a positive multiple of x.
    """
    j = (-m - 2) % x

    return BitArray(bin="1") + BitArray(bin="0" * j) + BitArray(bin="1")
