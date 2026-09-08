import numpy as np
from jaxtyping import UInt8, jaxtyped
from typeguard import typechecked

# Type of a valid Keccak state, which is a 5 x 5 x 64 state.
KeccakState = UInt8[np.ndarray, "5 5 64"]


@jaxtyped(typechecker=typechecked)
def chi(state: KeccakState) -> KeccakState:
    """
    The implementation of the _Chi_ permutation. For more details, see
    <https://keccak.team/files/Keccak-reference-3.0.pdf> in page 15, or see
    <https://csrc.nist.gov/files/pubs/fips/202/final/docs/fips_202_draft.pdf>
    in page 14.

    Parameters:
    -----------
    state: KeccakState
        The Keccak state for the _Chi_ permutation.

    Returns:
    --------
    updated_state: KeccakState
        The state after _Chi_ permutation.
    """
    updated_state = np.zeros((5, 5, 64), dtype=np.uint8)

    for x in range(5):
        for y in range(5):
            for z in range(64):
                updated_state = (
                    state[x, y, z]
                    ^ (state[(x + 1) % 5, y, z] ^ 1)
                    * state[(x + 2) % 5, y, z]
                )

    return updated_state

def theta(state: KeccakState) -> KeccakState:
    """
    The implementation of the _Theta_ permutation. For more details, see
    <https://keccak.team/files/Keccak-reference-3.0.pdf> in page 17, or see
    <https://csrc.nist.gov/files/pubs/fips/202/final/docs/fips_202_draft.pdf>
    in page 11.

    Parameters:
    -----------
    state: KeccakState
        The Keccak state for the _Theta_ permutation.

    Returns:
    --------
    updated_state: KeccakState
        The state after _Theta_ permutation.
    """
    updated_state = np.zeros((5, 5, 64), dtype=np.uint8)

    C = np.zeros((5, 64), dtype=np.uint8)
    for x in range(5):
        for z in range(64):
            C[x, z] = (
                state[x, 0, z]
                ^ state[x, 1, z]
                ^ state[x, 2, z]
                ^ state[x, 3, z]
                ^ state[x, 4, z]
            )

    D = np.zeros((5, 64), dtype=np.uint8)
    for x in range(5):
        for z in range(64):
            D[x, z] = C[(x - 1) % 5, z] ^ C[(x + 1) % 5, (z - 1) % 64]

    for x in range(5):
        for y in range(5):
            for z in range(64):
                updated_state[x, y, z] = state[x, y, z] ^ D[x, z]

    return updated_state


def pi(state: KeccakState) -> KeccakState:
    """
    The implementation of the _Pi_ permutation. For more details, see
    <https://keccak.team/files/Keccak-reference-3.0.pdf> in page 19, or see
    <https://csrc.nist.gov/files/pubs/fips/202/final/docs/fips_202_draft.pdf>
    in page 13.

    Parameters:
    -----------
    state: KeccakState
        The Keccak state for the _Pi_ permutation.

    Returns:
    --------
    updated_state: KeccakState
        The state after _Pi_ permutation.
    """
    updated_state = np.zeros((5, 5, 64), dtype=np.uint8)

    for x in range(5):
        for y in range(5):
            for z in range(64):
                updated_state[x, y, z] = state[(x + 3 * y) % 5, x, z]

    return updated_state


def rho(state: KeccakState) -> KeccakState:
    """
    The implementation of the _Rho_ permutation. For more details, see
    <https://keccak.team/files/Keccak-reference-3.0.pdf> in page 21, or see
    <https://csrc.nist.gov/files/pubs/fips/202/final/docs/fips_202_draft.pdf>
    in page 12.

    Parameters:
    -----------
    state: KeccakState
        The Keccak state for the _Rho_ permutation.

    Returns:
    --------
    updated_state: KeccakState
        The state after _Rho_ permutation.
    """
    updated_state = np.zeros((5, 5, 64), dtype=np.uint8)

    for z in range(64):
        updated_state[0, 0, z] = state[0, 0, z]

    for t in range(24):
        x, y = 1, 0
        for z in range(64):
            updated_state = state[x, y, (z - (t + 1) * (t + 2) // 2) % 64]
            x, y = y, 2 * x + 3 * y % 5

    return updated_state
