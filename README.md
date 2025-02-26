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

## Installation

### Prerequisites

JADE requires Python 3.8 (or higher) with the following packages:

```bash
# Essential packages
numpy
scipy
astropy
matplotlib
pandas
jitcode
jitcxde_common
symengine
emcee
corner
arviz
pathos
mcint

# Standard library packages (already included with Python)
# os, sys, math, multiprocessing, bisect, functools, time, warnings, itertools, argparse, zipfile
```

### Installing JADE

1. Clone the repository:
   ```bash
   git clone https://github.com/JADE-Exoplanets/JADE.git
   cd JADE
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Basic Usage

For a comprehensive introduction to JADE, please refer to the first tutorial:
[Tutorial 1: Basic Setup and Simulation](tutorials/tutorial1_fd1_fa3_fc_fa1.ipynb)

Once you are familiar with it, you can engage in more advanced use cases of JADE, described by the other [tutorials](tutorials/).

## Citation

If you use JADE in your research, please cite the follow article in your publication:

```
Attia, M., Bourrier, V., Eggenberger, P., et al. 2021, A&A, 647, A40
```

If you need to reference this repository, please refer to the [CITATION](CITATION.cff) file.

## Contributing

Contributions to JADE are welcome! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more information on how to get involved.

## License

JADE is licensed under the BSD 3-Clause License—see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This work has been carried out within the framework of the NCCR PlanetS supported by the Swiss National Science Foundation under grants 51NF40_182901 and 51NF40_205606. This project has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (project SPICE DUNE, grant agreement no. 947634; grant agreement no. 730890).
- The main contributors to the development of this software are M. Attia, V. Bourrier, J.-B. Delisle, and V. Ruelle.