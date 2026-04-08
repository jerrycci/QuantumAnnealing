## Compal GPU Annealer (CGA) Usage & Installation

The **CGA** folder contains example programs demonstrating the implementation of the Compal GPU Annealer across different API versions.

### Example Programs
* **Example 1: Traveling Salesperson Problem**
    * `tsp-demo.py`: Demonstrates usage of the **CGA 1.0 API**.
* **Example 2: Max-Cut Problem**
    * `maxcut_sample.py`: Demonstrates usage of the **CGA 2.0 API**, including support for the `init_spin` variable.

---

### Installation Guide (CGA 2.0)
The CGA 2.0 package is located in the `./CGA2.0-release` folder. Follow these steps to upgrade your environment:

#### Step 1: Extract the Package
Download and unzip the CGA 2.0 package file.
```bash
7z x cga2.0-release.7z -p"password"
```

#### Step 2: Remove Previous Version
Uninstall the existing version of the solver to avoid conflicts.
```bash
pip uninstall compal_solver-nstc
```

#### Step 3: Install CGA 2.0
Install the newly extracted wheel file.
```bash
pip install ./compal_solver_nstc-2.0.0-py3-none-any.whl
```

---

### Supported GPU Architectures
This library is compiled with support for the following NVIDIA GPU architectures and specific hardware models:

| Compute Capability | Architecture | Notable GPU Models Supported |
| :--- | :--- | :--- |
| **sm_70** | **Volta** | Tesla V100, Titan V |
| **sm_80** | **Ampere (DC)** | A100 (SXM4/PCIe) |
| **sm_86** | **Ampere (Consumer)** | RTX 3090, 3080, 3070, A6000, A40 |
| **sm_90** | **Hopper** | H100, H200, H800 |
| **sm_100** | **Blackwell (DC)** | B100, B200, GB200 Superchip |
| **sm_120** | **Blackwell (Consumer)** | RTX 5090, RTX 5080 |

