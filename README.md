# Design of Experiments for Circuit Optimization

## Description

The Design of Experiments for Circuit Optimization project is a Python program that uses design of experiments to study how three circuit parameters influence the propagation delay of a hypothetical circuit. It generates a factorial experimental design, evaluates a simulated delay at each design point, fits regression models with main effects and interaction effects, assesses those effects with ANOVA, plots main effects and interactions, estimates a response surface, identifies a promising configuration within the tested bounds, and validates predictions against additional simulated points that were not used to fit the model.

The program models a simple driver circuit charging a load. The three controllable factors are series resistance, load capacitance, and supply voltage. The propagation delay is calculated with the documented response equation:

```
R_on(V) = 400 / (V - 0.5)^1.3            driver on-resistance (ohm)
C_J(V)  = 0.4 / sqrt(1 + V / 0.7)         driver parasitic capacitance (pF)
t_pd    = ln(2) * (R + R_on(V)) * (C + C_J(V))     propagation delay (ps)
E_sw    = (C + C_J(V)) * V^2               switching energy (pJ)
```

The program is organized as a package of modules, each responsible for one part of the study. `circuit_model.py` contains the response equation and noise model, `design.py` builds the experimental designs, `analysis.py` performs the regression, ANOVA, and validation calculations, and `plots.py` creates the figures. `run_study.py` acts as the driver that runs each stage in order and writes the results to a report.

## Requirements

* Python 3
* pandas
* NumPy
* SciPy
* statsmodels
* Matplotlib
* pytest (for running the tests)

The required libraries can be installed using:

```
pip install -r requirements.txt
```

## Build and Run

Run the program from the project directory using:

```
python run_study.py
```

Depending on the Python installation, the following command may also be used:

```
python3 run_study.py
```

The tests can be run using:

```
pytest
```

## Usage

Run the program to perform the full study. The program generates its own simulated data and does not require user input. A different random seed or output folder can be chosen with `python run_study.py --seed 7 --out outputs_seed7`.

The program prints the report and saves it to `outputs/report.md`, along with CSV files of the simulated runs and analysis tables, and the figures `main_effects.png`, `interaction_effects.png`, `effect_estimates.png`, `response_surface.png`, and `validation.png`.

For example:

```
# DOE study report: hypothetical RC-driver propagation delay

## 1. Factors and response
        column unit  low (-1)  center (0)  high (+1)
code
xR       R_ohm  ohm       100         300        500
xC        C_pF   pF         1           3          5
xV    V_supply    V       1.2        2.25        3.3

Hand check: t_pd(100 ohm, 1 pF, 3.3 V) = 165.79 ps.

## 2. Factorial study (2^3, 2 replicates, + 4 centre points = 20 runs)
Model: delay_ps ~ xR + xC + xV + xR:xC + xR:xV + xC:xV

### Effects (change in delay from low to high level)
       change_ps   p_value  statistically_significant  practically_significant
xR           934 3.012e-13                       True                     True
xC         1,866 5.979e-16                       True                     True
xV        -1,218 2.779e-14                       True                     True
xR:xC      583.5 2.049e-11                       True                     True
xR:xV     -27.42   0.09522                      False                    False
xC:xV     -738.2  2.49e-12                       True                     True

### Curvature check
Mean of corner runs = 1511.5 ps, mean of centre runs = 1093.9 ps, difference = 417.6 ps (t = 25.4, p = 1.1e-09).
...
```

The study runs in stages. It first evaluates a 2^3 full factorial design with two replicates and four centre points, using the low and high settings shown above. The factorial regression is fitted in coded units, where each factor's low setting is -1 and its high setting is +1, so each coefficient is half of the change in delay from the low to the high setting. Because the centre points show significant curvature, the design is then extended with six axial points into a central composite design, and a quadratic response surface is fitted to the logarithm of the delay. The program searches this surface for the configuration with the lowest predicted delay that keeps switching energy at or below 6 pJ, and then compares its predictions against 30 randomly placed points that were not used to fit the model. The report calculates all results from the simulated runs rather than using hardcoded calculated values.

The simulated delay includes noise so that it behaves like a measured result. For each run, the actual resistance and capacitance are drawn from normal distributions around their nominal settings with standard deviations of 1% and 2%, mimicking component tolerance, and normally distributed measurement noise with a standard deviation of 10 ps is added to the delay. The random numbers are generated with a fixed seed so that results are reproducible, and each run stores both the noisy delay and the noise-free delay.

The report distinguishes statistical significance from practical engineering significance. An effect is statistically significant when its p-value is below 0.05, meaning it can be distinguished from noise given the number of runs. An effect is practically significant when it changes the delay by 50 ps or more, which is used here as a hypothetical threshold for an effect large enough to change a design decision. The R x V interaction shows the difference between the two: it is not detected in the 20-run study, but when the factorial is repeated 200 times it becomes statistically significant while remaining about 13 ps, well below the practical threshold.

## Hypothetical Simulator and Real Circuits

All results come from the hypothetical response equation, not from a real circuit simulation tool. The study can show how the design of experiments workflow behaves and which effects matter within this equation. Conclusions about actual delay values, a real optimum supply voltage, or the behaviour of a specific process or part would require a real circuit simulation tool, such as SPICE with device models, or measurements from hardware. The equation leaves out effects such as input slew, temperature, process corners, interconnect, and correlated variation, and its device constants were chosen for illustration.

## AI Disclosure

The data used in this project was generated using AI. The concepts used in this project were explained to me by AI.
