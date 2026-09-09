import numpy as np

from permutations import STATE_SHAPE, chi, pi, rho, theta


def test_permotions():
    state = np.zeros(STATE_SHAPE, dtype=np.uint8)
    state[0, 0, 0] = 1

    for _ in range(2):
        state = chi(pi(rho(theta(state))))
    state = pi(rho(theta(state)))

    for x in range(5):
        for y in range(2):
            for z in range(64):
                if state[x, y, z] == 1:
                    print(f"{x}, {y}, {z}")


if __name__ == "__main__":
    test_permotions()
