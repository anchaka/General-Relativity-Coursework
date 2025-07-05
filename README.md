# General-Relativity-Coursework
Modelled and visualised black hole mergers producing gravitational waves. Utilised the Leapfrog algorithm and Post Newtonian Correction for Numerical modelling in Python 
# Numerical Integration of Black Hole Binaries with Post-Newtonian Corrections

**Module:** General Relativity  
**Institution:** University of Surrey  
**Assessment:** Coursework 2022  
**Author (URN):** Anchaka Ghatge  

---

## Overview

This project implements a numerical simulation of binary black hole systems with relativistic corrections. Two scenarios are analysed:

1. **Eccentric Binary**  
2. **Circular Binary**

Using a **Leapfrog integrator with adaptive time-step**, the code models:
- Relativistic perihelion precession (PN1 and PN2)
- Orbital decay from gravitational wave emission (PN2.5)

---

## Contents

| File                             | Description                                      |
|----------------------------------|-------------------------------------------------|
| `blackhole_binary_ecc_LSA.py`    | Simulation for **eccentric** black hole binary  |
| `blackhole_binary_circ_LSA.py`   | Simulation for **circular** black hole binary   |
| `binary.ecc.init`                | Initial conditions for eccentric binary         |
| `binary.circ.init`               | Initial conditions for circular binary          |
| `binary.ecc.orb`                 | Orbital data over time (eccentric binary)       |
| `binary.circ.orb`                | Orbital data over time (circular binary)        |
| `plots/`                         | Generated plots and figures                    |
| `report.pdf`                     | Coursework report                              |
| `README.md`                      | This file                                      |

---

## Requirements

- **Python 3.x**
- Python libraries:
  - `numpy`
  - `matplotlib`
  - `scipy`

Install dependencies:

```bash
pip install numpy matplotlib scipy

Running the Code

Eccentric Binary
python blackhole_binary_ecc_LSA.py
Circular Binary
python blackhole_binary_circ_LSA.py

Output:

Initial units and constants printed to screen
Precession rates (theoretical & numerical)
Orbital evolution written to binary.ecc.orb and binary.circ.orb
Plots saved in plots/

Features

Leapfrog integration with adaptive time-step
Post-Newtonian corrections up to 2.5PN
Computes and compares numerical and theoretical:
Pericentre advance
Orbital decay (Peters equations)

Generates publication-quality plots of:
Orbits
Semi-major axis a(t) vs time
Eccentricity e(t) vs time
a(t) vs aₚ(t) for circular binaries

Limitations

Relatively long runtimes for 400-year integrations
Only up to 2.5PN corrections included
No parallelization

References

Peters, P. C., Gravitational Radiation and the Motion of Two Point Masses, Phys. Rev., 1964
Maggiore, M., Gravitational Waves Vol. 1
PHYM053 Lecture Notes (University of Surrey)

