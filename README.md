# Keccak-f[1600] Bit Diffusion Analysis

A bit-level implementation and empirical evaluation of the **Keccak-f[1600]**
state permutation engine, designed to analyze the Strict Avalanche Criterion
(SAC) and bit diffusion behavior across state planes.

To ensure complete mathematical correctness, the core engine is validated
against the same algorithm in `hashlib`.

For detailed theoretical derivations, step mapping equations, and API specs, see 
[design.md](design.md).

## Core Objective & Methodology

The primary goal of this project is to quantitatively measure the active bit
expansion (Hamming weight) resulting from single-bit input perturbations.

* **Perturbation Strategy**: Injects a single active bit at $(x, y, z)$ across
  representative lane coordinates $(x, y)$ for all $z \in [0, 63]$.
* **Permutation Pipeline**: Runs 2.75 rounds of state transformations
  ($\chi \circ \pi \circ \rho \circ \theta$).
* **Metric**: Evaluates state-wide diffusion density by counting active bits
  ($1$-bits) in the lower state planes ($y \in \{0, 1\}$).

---

## Quick Start

### Prerequisites

* **Python**: 3.10 or higher
* **Git**: Installed and configured

### 1. Requirements & Setup

* Clone the repository:

```bash
git clone https://github.com/seeleseelebronya/keccak.git
```

* Create python virtual environment:
  * **In project directory**:
  * Linux/macOS:
  ```bash
  python -m venv .venv
  source ./.venv/bin/activate
  ```
  * Windows (PowerShell):
  ```PowerShell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```

* Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run Diffusion Analysis (influence.py)
Execute the influence experiment to measure active bit propagation:

```bash
python influence.py
```

Output: Generates `influence.csv` containing active bit counts for evaluated
state coordinates.

### *3. Verify Permutation Engine (`sha3_256.py`)

Run the test script:

```bash
pytest test_sha_256.py
```
