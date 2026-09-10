"""The main program to test the influence of a bit.
"""

import csv
import numpy as np
from permutations import STATE_SHAPE, STATE_Z, chi, pi, rho, theta

def main():
    """The main function to test the influence and out put a CSV file.
    """
    with open("influences.csv", "w", newline="") as csvfile:
        fieldnames = ["position", "influence"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for z in range(STATE_Z):
            writer.writerow({"position": z, "influence": influence(z)})


def influence(z: int) -> int:
    """The function to test the influence of the position (0, 0, z).

    With the given `z`, set `state[0, 0, z] = 1`. Then runs two round of
    permutations θ, ρ, π, χ and one round permutations θ, ρ and π. Count
    the positions that be set to 1 in the first two layers

    (`0 <= x < 5, 0 <= y < 2, 0 <= z < w`).

    Parameters:
    -----------
    z: int
        The position of z, 0 <= z < w.

    Returns:
    --------
    count: int
        The count of active bits (influence).
    """
    state = np.zeros(STATE_SHAPE, dtype=np.uint8)
    state[0, 0, z] = 1

    for _ in range(2):
        state = chi(pi(rho(theta(state))))
    state = pi(rho(theta(state)))

    count = 0
    for r in range(5):
        for s in range(2):
            for t in range(STATE_Z):
                if state[r, s, t] == 1:
                    count += 1

    return count


if __name__ == "__main__":
    main()
