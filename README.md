# MHGphormer: Metapath-based Heterogeneous Graph-Transformer Network

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This is the official PyTorch implementation of the **Metapath-based Heterogeneous Graph-Transformer Network (MHGphormer)**. The architecture and its application are proposed in our paper:

> **["Joint Spectrum, Precoding, and Phase Shifts Design for RIS-Aided Multiuser MIMO THz Systems"](https://people.ece.ubc.ca/vincentw/J/MW-TCOM-2024.pdf)**  
> Accepted for publication in *IEEE Transactions on Communications*, 2024.

MHGphormer leverages advanced heterogeneous graph transformer architectures to optimize joint spectrum, precoding, and phase shifts for Reconfigurable Intelligent Surface (RIS)-aided multiuser MIMO systems in the Terahertz (THz) band.

---

## 📖 Table of Contents
- [Architecture Overview](#-architecture-overview)
- [Repository Structure](#-repository-structure)
- [Requirements & Installation](#-requirements--installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Citation](#-citation)
- [Contact](#-contact)

---

## 🧠 Architecture Overview

The system models the complex interactions between Users, the Base Station (BS), and the RIS as a heterogeneous graph. MHGphormer captures these interactions through multiple carefully designed metapaths and aggregates them via a novel Transformer-based attention mechanism.

Key components optimized by the network:
1. **Precoding Matrix (Beamforming):** Formulated at the BS to maximize signal power directed toward users while minimizing inter-user interference.
2. **RIS Phase Shifts:** Optimized via complex-valued layers to properly reflect and align incoming THz signals.
3. **Sub-band Allocation:** Predicts optimal bandwidth allocations for different users using custom constrained multi-layer perceptrons.

---

## 🗂 Repository Structure

The codebase has been refactored for modularity, readability, and scalability.

```text
MHGphormer/
│
├── src/
│   ├── __init__.py
│   ├── config.py           # Global hyperparameters, system setup & bounds
│   ├── dataset.py          # Data generation and Dataloader utilities
│   ├── trainer.py          # Model training, validation loop, and custom loss formulation
│   └── models/             # Neural network definitions
│       ├── __init__.py
│       ├── layers.py       # Custom MLPs, Complex MLPs, and Transformer building blocks
│       └── mhgphormer.py   # Main SeHGNN architecture
│
├── main.py                 # Execution entry point
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 🛠 Requirements & Installation

### Prerequisites

Ensure you have Python 3.8+ and a CUDA-capable GPU. The implementation heavily relies on PyTorch's CUDA backend for accelerated tensor computations.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ali-Meh619/MHGphormer.git
   cd MHGphormer
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

You can train and evaluate the model using the provided `main.py` entry point. It automatically orchestrates dataset generation, model initialization, training loop, and final evaluation.

```bash
python main.py
```

### What happens during execution?
1. **Dataset Generation (`src/dataset.py`):** Randomizes user locations and calculates corresponding THz channel gains, delays, and path losses. Converts system graphs into heterogeneous meta-paths.
2. **Model Initialization (`src/models/mhgphormer.py`):** Instantiates the MHGphormer network.
3. **Training (`src/trainer.py`):** Starts the epoch loop. During each iteration, the model outputs beamforming vectors, phase shifts, and bandwidth allocation. The physical layer optimization problem is directly embedded into the loss function.
4. **Evaluation:** Outputs the final achievable sum-rate over the test dataset and saves the optimized model checkpoint in the `RIS-MIMO-THz` directory.

> **⚠️ Memory Note:** The original default parameters (`batch=250`, `int_samp=100000`) require upwards of 80GB of GPU VRAM due to the massive continuous tensor allocations for channel integration. If you encounter `CUDA Out of Memory` errors, please lower `int_samp` (e.g., `10000`) or reduce the `batch` size in `src/config.py`.

---

## ⚙️ Configuration

Hyperparameters, network dimensions, and system assumptions (like transmission power, noise density, frequencies, etc.) are centralized in `src/config.py`. 

To experiment with different RIS sizes, BS antennas, or learning rates, simply modify the `ARGS` dictionary inside `src/config.py`:

```python
ARGS = {
    "IRS_elements": 64,         # Number of reflecting elements on the RIS
    "BS_antenna": 32,           # Number of Base Station antennas
    "num_users": 6,             # Number of simultaneous users
    "epochs": 250,              # Training iterations
    "lr_init": 0.0005,          # Initial learning rate
    ...
}
```

---

## 📜 Citation

If you find our paper or this codebase useful in your research, please kindly cite our paper:

```bibtex
@article{b16,
  title={Joint spectrum, precoding, and phase shifts design for {RIS}-aided multiuser {MIMO} {TH}z systems},
  author={Mehrabian, Ali and Wong, Vincent W. S.},
  journal={IEEE Trans. Commun.},
  volume={72},
  number={8},
  pages={5087-5101},
  month={Aug.},
  year={2024}
}
```

---

## 📬 Contact

For any questions, discussions, or bug reports, please feel free to reach out:
- **Ali Mehrabian:** alimehrabian619@ece.ubc.ca | alimehrabian619@yahoo.com
