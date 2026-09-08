import numpy as np
from jaxtyping import UInt8, jaxtyped
from typeguard import typechecked

# Type of a valid Keccak state, which is a 5 x 5 x 64 state.
KeccakState = UInt8[np.ndarray, "5 5 64"]


@jaxtyped(typechecker=typechecked)
def chi(state: KeccakState) -> KeccakState:
    """
    The implementation of the _Chi_ permutation. For more details, see
    <https://keccak.team/files/Keccak-reference-3.0.pdf> page 15.

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

    for z in range(64):
        for y in range(5):
            row_x = state[:, y, z]
            row_x_plus_1 = state[[1, 2, 3, 4, 0], y, z]
            row_x_plus_2 = state[[2, 3, 4, 0, 1], y, z]

            updated_state[:, y, z] = row_x ^ ((1 ^ row_x_plus_1) & row_x_plus_2)

    return updated_state

def theta():
    ...


def pi():
    ...


def rho():
    ...
