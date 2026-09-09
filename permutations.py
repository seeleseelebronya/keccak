"""Keccak-f[1600] State Permutations.

This module provides bit-level, explicit implementations of the five core
state transformations (Theta, Rho, Pi, Chi, Iota) specified in NIST FIPS 202
(Link: <https://csrc.nist.gov/pubs/fips/202/final>).

State representation:
    A 3D NumPy array of shape (5, 5, 64) with `np.uint8` dtype, where each
    element represents a single bit (0 or 1).
"""

import numpy as np
from jaxtyping import UInt8, jaxtyped
from typeguard import typechecked

# Length of rows
STATE_X = 5

# Length of columns
STATE_Y = 5

# Length of lanes (also w in NIST FIPS 202)
STATE_Z = 64

# Shape of a state, which is 5x5x64
STATE_SHAPE = (STATE_X, STATE_Y, STATE_Z)

# L satisfies STATE_Z = 2 ^ L
L = int(np.log2(STATE_Z))

# Type of a valid Keccak state, which is a 5x5x64 state cube
KeccakState = UInt8[np.ndarray, f"{STATE_X} {STATE_Y} {STATE_Z}"]


@typechecked
def chi(state: KeccakState) -> KeccakState:
    """The Chi (χ) permutation.

    Parameters:
    -----------
    state: KeccakState
        The current 5x5x64 Keccak state cube.

    Returns:
    --------
    updated_state: KeccakState
        The updated 5x5x64 Keccak state cube after Chi transformation.
    """
    updated_state = np.zeros(STATE_SHAPE, dtype=np.uint8)

    for x in range(STATE_X):
        for y in range(STATE_Y):
            for z in range(STATE_Z):
                updated_state[x, y, z] = (
                    state[x, y, z]
                    ^ (state[(x + 1) % 5, y, z] ^ 1)
                    * state[(x + 2) % 5, y, z]
                )

    return updated_state

@typechecked
def theta(state: KeccakState) -> KeccakState:
    """The Theta (θ) permutation.

    Parameters:
    -----------
    state: KeccakState
        The current 5x5x64 Keccak state cube.

    Returns:
    --------
    updated_state: KeccakState
        The updated 5x5x64 Keccak state cube after Chi transformation.
    """
    updated_state = np.zeros(STATE_SHAPE, dtype=np.uint8)

    C = np.zeros((5, 64), dtype=np.uint8)
    for x in range(STATE_X):
        for z in range(STATE_Z):
            C[x, z] = (
                state[x, 0, z]
                ^ state[x, 1, z]
                ^ state[x, 2, z]
                ^ state[x, 3, z]
                ^ state[x, 4, z]
            )

    D = np.zeros((5, 64), dtype=np.uint8)
    for x in range(STATE_X):
        for z in range(STATE_Z):
            D[x, z] = C[(x - 1) % 5, z] ^ C[(x + 1) % 5, (z - 1) % 64]

    for x in range(STATE_X):
        for y in range(STATE_Y):
            for z in range(STATE_Z):
                updated_state[x, y, z] = state[x, y, z] ^ D[x, z]

    return updated_state


@typechecked
def pi(state: KeccakState) -> KeccakState:
    """The Pi (π) permutation.

    Parameters:
    -----------
    state: KeccakState
        The current 5x5x64 Keccak state cube.

    Returns:
    --------
    updated_state: KeccakState
        The updated 5x5x64 Keccak state cube after Pi transformation.
    """
    updated_state = np.zeros(STATE_SHAPE, dtype=np.uint8)

    for x in range(STATE_X):
        for y in range(STATE_Y):
            for z in range(STATE_Z):
                updated_state[x, y, z] = state[(x + 3 * y) % 5, x, z]

    return updated_state


@typechecked
def rho(state: KeccakState) -> KeccakState:
    """The Rho (ρ) permutation.

    Parameters:
    -----------
    state: KeccakState
        The current 5x5x64 Keccak state cube.

    Returns:
    --------
    updated_state: KeccakState
        The updated 5x5x64 Keccak state cube after Rho transformation.
    """
    updated_state = np.zeros(STATE_SHAPE, dtype=np.uint8)

    for z in range(STATE_Z):
        updated_state[0, 0, z] = state[0, 0, z]

    x, y = 1, 0
    for t in range(24):
        for z in range(STATE_Z):
            updated_state[x, y, z] = state[
                x, y, (z - (t + 1) * (t + 2) // 2) % 64
            ]
            x, y = y, (2 * x + 3 * y) % 5

    return updated_state


@typechecked
def iota(state: KeccakState, round_index: int) -> KeccakState:
    """The Iota (ι) permutation.

    Parameters:
    -----------
    state: KeccakState
        The current 5x5x64 Keccak state cube.

    Returns:
    --------
    updated_state: KeccakState
        The updated 5x5x64 Keccak state cube after Iota transformation.
    """
    updated_state = np.zeros(STATE_SHAPE, dtype=np.uint8)

    for x in range(STATE_X):
        for y in range(STATE_Y):
            for z in range(STATE_Z):
                updated_state[x, y, z] = state[x, y, z]

    rc = np.zeros(64, dtype=np.uint8)
    for j in range(L):
        rc[2 ** j - 1] = _rc(j + 7 * round_index)

    for z in range(STATE_Z):
        updated_state = state[0, 0, z] ^ rc[z]

    return updated_state


def _rc(t: int) -> int:
    """A helper function for `iota()` permutation.

    Parameters:
    -----------
    t: integer

    Returns:
    rc(t): int, actually a bit
    """
    mod = t % 255
    if mod == 0:
        return 1

    r = [1, 0, 0, 0, 0, 0, 0, 0]
    for _ in range(mod):
       r.insert(0, 0)

       r[0] = r[0] + r[8]
       r[4] = r[4] + r[8]
       r[5] = r[5] + r[8]
       r[6] = r[6] + r[8]

       r = r[:8]

    return r[0]
