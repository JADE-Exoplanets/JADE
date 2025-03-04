# JADE: Joining Atmosphere and Dynamics for Exoplanets

JADE is a comprehensive Python framework for simulating the coupled evolution of exoplanetary atmospheres and orbital dynamics in hierarchical three-body systems.

## Overview

JADE couples the most relevant mechanisms affecting close-orbiting exoplanets:

### Dynamical Components:
- Simulates the evolution of hierarchical three-body systems
- Models von Zeipel–Lidov–Kozai resonances using secular formalism
- Truncates the Hamiltonian up to the 4th (hexadecapolar) order
- Integrates short-range forces dampening resonances:
  - Tidal dissipation
  - Rotational dissipation
  - General relativistic precession

### Atmospheric Components:
- Models planets as extended objects with complex structure:
  - Iron nucleus
  - Silicate mantle
  - H/He atmosphere with trace metals
- Dynamically integrates atmospheric evolution in response to:
  - Orbital changes
  - Stellar radiative input
  - Internal heating
- Simulates XUV-induced photo-evaporation

All these components interact holistically, making JADE a unique tool for studying the long-term evolution of exoplanetary systems over Gyr timescales.

## Installation Guide

First, clone the JADE repository:
   ```bash
   git clone https://github.com/JADE-Exoplanets/JADE.git
   cd JADE
   ```

Then, you have to set up a virtual environment. JADE performs computationally intensive simulations that require specific optimizations for best performance. We provide multiple installation methods tailored to different hardware configurations. **Be sure to activate the virtual environment you install before running any JADE simulation.**

### Automated Setup (Recommended)

Our setup script automatically detects your hardware and creates an optimized environment:

   ```bash
   # Run the script with default settings (auto-detects best method)
   python setup.py
   ```

The script will:
1. Detect your CPU architecture
2. Choose the best installation method for your system
3. Create an optimized environment
4. Run a performance benchmark

#### Available Installation Methods

Running the script with no argument makes it automatically detect the best method. You can also impose a method using the `--method` argument. The script supports three installation methods, each with different advantages:

| Method | Best For | Advantages | Command |
|--------|----------|------------|---------|
| **conda** | Intel CPUs | MKL optimizations, 10–250× faster | `--method conda` |
| **UV** | Apple Silicon, AMD | Guarantees Python 3.8, fast install | `--method uv` |
| **pip** | Fallback option | Standard Python tooling | `--method pip` |

#### Example Usage

Aside from `--method`, the setup script also accepts the `--env` argument, to choose the name of your virtual environment, as well as the `--force` argument, to force the recreation of an existing environment. Here are some example usages:

   ```bash
   # For Intel systems with conda already installed
   python setup.py --method conda --env jade
   
   # For M1/M2 Macs or AMD systems (ensures Python 3.8)
   python setup.py --method uv
   
   # Force recreation of an existing environment
   python setup.py --force
   ```

### Manual Installation Options

If you prefer to set up your environment manually, follow one of these approaches:

#### Option 1: Using conda (Best for Intel CPUs)

   ```bash
   # Create conda environment from provided configuration
   conda env create -f environment.yml
   conda activate jade
   ```

#### Option 2: Using UV (Recommended for Apple Silicon/AMD)

   ```bash
   # Install UV if not already available
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Install Python 3.8 and create environment
   uv python install cpython-3.8
   uv venv --python cpython-3.8
   source .venv/bin/activate
   
   # Install dependencies
   uv pip install -r requirements.txt
   ```

#### Option 3: Standard pip/venv (Not recommended for large simulations)

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Linux/Mac
   # or
   # .venv\Scripts\activate  # On Windows
   
   pip install -r requirements.txt
   ```

### Performance Expectations

Performance varies significantly based on installation method and hardware:

- **Intel CPUs with conda (MKL), or Apple Silicon/AMD with UV (OpenBLAS)**: A standard JADE simulation (e.g., `input/examples/example_fd1.txt`) may complete in seconds
- **Other configurations or standard pip**: The same simulation could take 10–250× longer

The difference is primarily due to optimized linear algebra libraries (MKL/OpenBLAS) in numerical packages, which dramatically impact the integration of differential equations.

### Troubleshooting

If you experience slow performance:

1. **Verify installation method**: Use `conda` for Intel systems and `uv` for others
2. **Check Python version**: JADE requires Python 3.8
3. **Run the benchmark**: The setup script includes a benchmark to verify performance
4. **Memory requirements**: Ensure you have sufficient RAM (4GB minimum)

## Basic Usage

For a comprehensive introduction to JADE, please refer to the first tutorial:
[Tutorial 1: Basic Setup and Simulation](tutorials/tutorial1_fd1_fa3_fc_fa1.ipynb)

Once you are familiar with it, you can engage in more advanced use cases of JADE, described by the other [tutorials](tutorials/).

## Citation

If you use JADE in your research, please cite the following article in your publication:
```
Attia, M., Bourrier, V., Eggenberger, P., et al. 2021, A&A, 647, A40
```

If you need to reference this repository, please refer to the [CITATION](CITATION.cff) file.

## Contributing

Contributions to JADE are welcome! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more information on how to get involved.

## License

JADE is licensed under the BSD 3-Clause License—see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The main contributor to the development of this software is Mara Attia, with help from Jean-Baptiste Delisle and Victor Ruelle. We acknowledge the use of the Claude AI assistant (Anthropic, 2024) for code optimization.
- We offer our thanks to Elsa Bersier (ESBDI, Geneva) for designing the JADE logo.
- This work has been carried out within the framework of the NCCR PlanetS supported by the Swiss National Science Foundation under grants 51NF40_182901 and 51NF40_205606. This project has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (project SPICE DUNE, grant agreement no. 947634; grant agreement no. 730890).
