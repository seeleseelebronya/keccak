"""The SHA3-256 implementation and some helper functions.

The functions are specified in NIST FIPS 202
(Link: <https://csrc.nist.gov/pubs/fips/202/final>)
"""

from collections.abc import Callable
import numpy as np
import permutations as pmt

def sha3_256(message: str) -> bytes:
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
    ...


def keccak_p(rounds: int, s: str,b: int=1600) -> str:
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

    s: str
        String of length b.

    b: int, optional
        The default is 1600, which is the width of a KECCAK-p
        permutation in bits.

    Returns:
    --------
    s_prime: str
        String of length b after the permutations.
    """
    ...


def sponge(pad: Callable[[int, int], list[int]], r: int, m: str, d: int) -> str:
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

    m: str
        The string of message.

    d: int, nonnegative

    Returns:
    --------
    z: str
        String of length `d`.
    """
    ...


def pad_10_asterisk_1(x: int, m: int):
    """ Given a positive integer `x` and a non-negative integer `m`, the
    output `pad(x, m)` is a string with the property that
    `m + len(pad(x, m))` is a positive multiple of x.

    Parameters:
    -----------
    x: int, positive
    m: int, nonnegative

    Returns:
    z: str
        String Z such that m + len(Z) is a positive multiple of x.
    """
