# Design Document

## References

* **NIST FIPS 202**:
  [SHA-3 Standard: Permutation-Based Hash and Extendable-Output Functions](https://csrc.nist.gov/pubs/fips/202/final)

## Project structure

### `influence.py`
The script that measures the diffusion capability of the
Keccak-f[1600] permutation over 2.75 rounds. It evaluates the impact of
single-bit perturbations across key state positions.

* **Target Positions**: Evaluates initial states with a single active bit at
  $(x, y, z)$ for all $z \in [0, 63]$ across six representative lane
  coordinates:

  $$(x, y) \in \{(0, 0), (2, 0), (0, 1), (2, 1), (0, 2), (2, 2)\}$$

* **Transformation Sequence**: For each active bit position, the state
undergoes two full rounds of $\chi \circ \pi \circ \rho \circ \theta$
followed by a partial round of $\pi \circ \rho \circ \theta$.
* **Metric**: Counts the total number of active bits (Hamming weight) in the
first two planes ($0 \le x < 5, 0 \le y < 2$) to assess state-wide active bit
expansion and structural symmetry.)

### `permutation.py`
The file implements the 5 permutations ($\theta$, $\rho$, $\pi$, $\chi$,
and $\iota$).

### `sha3_256.py`
The file implements SHA3-256.

### `test_sha3_256.py`
The script to test the correctness of the SHA3-256 implementation.

### `requirements.txt`
The required python libs.

## API References

### `influence.py`
* `influence(x: int, y: int, z: int) -> int`
  * Calculates the active bit count (Hamming weight) across the first
  two planes ($y \in \{0, 1\}$) after 2.75 permutation rounds for a perturbation
  at position $(x, y, z)$.

### `sha3_256.py`

Implements padding, sponge construction, and bitstream-to-state conversions.

* `sha3_256(message: bytes) -> str`
  * **Spec Reference**: Section 6.1 (SHA3-256 Specification)
  * **Main API**: Accepts raw message bytes, appends the SHA-3 domain
  separator (`01`), pads the sequence, executes the Keccak sponge pipeline, and
  returns the 256-bit digest as a hexadecimal string.
* `keccak_p(s: BitArray, b: int = 1600, rounds: int = 24) -> BitArray`
  * **Spec Reference**: Section 3.3, Algorithm 7 ($KECCAK\text{-}p[b, n_r]$)
  * Runs $n_r$ iterations of the round permutation
  $Rnd = \iota \circ \chi \circ \pi \circ \rho \circ \theta$ on a 1600-bit
  vector.
* `sponge(pad: Callable, r: int, m: BitArray, d: int) -> BitArray`
  * **Spec Reference**: Section 4 (Sponge Construction)
  * Absorbs $r$-bit blocks into the Keccak state and squeezes $d$ output bits.
* `pad_10_asterisk_1(x: int, m: int) -> BitArray`
  * **Spec Reference**: Section 5.1, Algorithm 9 ($pad10^*1$)
  * Multi-rate padding rule ($pad10^*1$) satisfying $m + |Z| \equiv 0 \pmod x$.

### `permutations.py`

Handles state transformations operating on a 3D NumPy array of shape
$(5, 5, 64)$ with `uint8` data type (`KeccakState`).

* `theta(state: KeccakState) -> KeccakState`
  * **Spec Reference**: Section 3.2.1, Algorithm 1 ($\theta$)
  * Computes parity bits for each column and applies column-wise XOR diffusion.
* `rho(state: KeccakState) -> KeccakState`
  * **Spec Reference**: Section 3.2.2, Algorithm 2 ($\rho$)
  * Performs triangular lane rotations along the $z$-axis for specified shifts.
* `pi(state: KeccakState) -> KeccakState`
  * **Spec Reference**: Section 3.2.3, Algorithm 3 ($\pi$)
  * Permutes the order of the 25 lanes within the state grid.
* `chi(state: KeccakState) -> KeccakState`
  * **Spec Reference**: Section 3.2.4, Algorithm 4 ($\chi$)
  * Applies non-linear bitwise operations across row elements ($x$-axis).
* `iota(state: KeccakState, round_index: int) -> KeccakState`
  * **Spec Reference**: Section 3.2.5, Algorithm 5 (`rc`) & Algorithm 6
    ($\iota$)
  * Injects round constants ($RC$) into lane $(0,0)$ to break rotational
    symmetry.
