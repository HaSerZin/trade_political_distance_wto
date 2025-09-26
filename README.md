### **`README.md`**

# A Quantitative Framework for Modeling Political Distance and Trade

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Type Checking: mypy](https://img.shields.io/badge/type_checking-mypy-blue)](http://mypy-lang.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2509.17303-b31b1b.svg)](https://arxiv.org/abs/2509.17303)
[![Year](https://img.shields.io/badge/Year-2025-purple)](https://github.com/chirindaopensource/trade_political_distance_wto)
[![Discipline](https://img.shields.io/badge/Discipline-International%20Political%20Economy-blue)](https://github.com/chirindaopensource/trade_political_distance_wto)
[![Methodology](https://img.shields.io/badge/Methodology-Structural%20Gravity%20%7C%20PPML%20%7C%20Particle%20Filter-orange)](https://github.com/chirindaopensource/trade_political_distance_wto)
[![Data Source](https://img.shields.io/badge/Data-GDELT%20%7C%20UNGA%20Votes%20%7C%20IMF%20DOTS-lightgrey)](https://github.com/chirindaopensource/trade_political_distance_wto)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%23025596?style=flat&logo=scipy&logoColor=white)](https://scipy.org/)
[![Statsmodels](https://img.shields.io/badge/statsmodels-1A568C.svg?style=flat)](https://www.statsmodels.org/stable/index.html)
[![PyFixest](https://img.shields.io/badge/pyfixest-0B559E.svg?style=flat)](https://github.com/s3alfisc/pyfixest)
[![Pydantic](https://img.shields.io/badge/pydantic-E92063.svg?style=flat)](https://pydantic-docs.helpmanual.io/)
[![PyYAML](https://img.shields.io/badge/PyYAML-4B5F6E.svg?style=flat)](https://pyyaml.org/)
[![Joblib](https://img.shields.io/badge/joblib-2F72A4.svg?style=flat)](https://joblib.readthedocs.io/en/latest/)
[![Modelsummary](https://img.shields.io/badge/modelsummary-D4352C.svg?style=flat)](https://modelsummary.com/)
[![Analysis](https://img.shields.io/badge/Analysis-Geopolitical%20Risk-brightgreen)](https://github.com/chirindaopensource/trade_political_distance_wto)
[![Framework](https://img.shields.io/badge/Framework-Bayesian%20Filtering-blueviolet)](https://github.com/chirindaopensource/trade_political_distance_wto)
[![Model](https://img.shields.io/badge/Model-Non--Linear%20Panel-red)](https://github.com/chirindaopensource/trade_political_distance_wto)
[![Validation](https://img.shields.io/badge/Validation-Temporal%20Analysis-yellow)](https://github.com/chirindaopensource/trade_political_distance_wto)
[![Robustness](https://img.shields.io/badge/Robustness-Sensitivity%20Analysis-lightgrey)](https://github.com/chirindaopensource/trade_political_distance_wto)

--

**Repository:** `https://github.com/chirindaopensource/trade_political_distance_wto`

**Owner:** 2025 Craig Chirinda (Open Source Projects)

This repository contains an **independent**, professional-grade Python implementation of the research methodology from the 2025 paper entitled **"Trade, Political Distance and the World Trade Organization"** by:

*   Samuel Hardwick

The project provides a complete, end-to-end computational framework for quantifying the economic penalty of political friction on international trade and assessing the mitigating role of the World Trade Organization. It delivers a modular, auditable, and extensible pipeline that replicates the paper's entire workflow: from rigorous data validation and the sophisticated filtering of high-frequency event data, through the estimation of complex non-linear panel models with high-dimensional fixed effects, to the final synthesis and presentation of results.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callable: execute_full_research_pipeline](#key-callable-execute_full_research_pipeline)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [Recommended Extensions](#recommended-extensions)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project provides a Python implementation of the methodologies presented in the 2025 paper "Trade, Political Distance and the World Trade Organization." The core of this repository is the iPython Notebook `trade_political_distance_wto_draft.ipynb`, which contains a comprehensive suite of functions to replicate the paper's findings, from initial data validation to the final generation of tables, figures, and a synthesis report.

The paper addresses a key question in international political economy: Do multilateral institutions like the WTO shield commerce from geopolitical tensions? This codebase operationalizes the paper's advanced approach, allowing users to:
-   Rigorously validate and assess the quality of a complex, multi-source panel dataset.
-   Extract a smoothed signal of political relations from noisy, high-frequency news event data using a state-space model (Particle Filter).
-   Estimate a state-of-the-art structural gravity model using Poisson Pseudo-Maximum Likelihood (PPML) with high-dimensional fixed effects.
-   Analyze the extensive margin of trade using a Logit model with advanced, bias-corrected estimation and inference (Split-Panel Jackknife with Bootstrap).
-   Quantify the economic and statistical significance of the WTO's mitigating effect on political distance.
-   Systematically test the stability of the findings across a wide array of robustness checks.

## Theoretical Background

The implemented methods are grounded in modern econometrics, computational statistics, and international political economy.

**1. Structural Gravity Model:**
The workhorse model is the structural gravity equation of trade, which has strong theoretical microfoundations. The model explains bilateral trade flows as a function of economic size and various trade frictions or promoters. In this paper, the key friction is political distance. The model is specified in its multiplicative form and estimated with PPML to handle heteroskedasticity and zero trade flows correctly. The inclusion of high-dimensional fixed effects (exporter-year, importer-year, and country-pair) is critical to control for all time-varying country-specific factors (Multilateral Resistance Terms) and all time-invariant dyadic factors, isolating the effect of interest.
$$ X_{ijt} = \exp\left(\left[\beta_0 PD_{ijt} + \sum_z \beta_z (PD_{ijt} \times z_{ijt})\right] \mathbf{1}_{(i \neq j)} + \delta_{it} + \delta_{jt} + \delta_{ij} + \text{Border}_{ijt}\right) \times \varepsilon_{ijt} $$

**2. State-Space Model and Particle Filter:**
To measure short-term political relations, the paper treats the "true" but unobserved political stance between two countries as a latent state ($s_{ijt}$) that evolves over time. The noisy, high-frequency GDELT event data ($y_{ijt}$) is treated as a measurement of this state.
-   **State Equation (Random Walk):** $s_{ijt} = s_{ij,t-1} + v_t$, where $v_t \sim \mathcal{N}(0, Q_{ij})$
-   **Observation Equation:** $y_{ijt} = s_{ijt} + \eta_t$, where $\eta_t \sim \mathcal{N}(0, R_{ijt})$
Because the distributions may not be strictly Gaussian, this system is solved using a **Particle Filter**, a Sequential Monte Carlo method that approximates the posterior distribution of the state at each time step. This is a sophisticated technique to filter signal from noise.

**3. Inference in Non-Linear Panel Models:**
-   **PPML:** Standard errors are clustered at the country-pair level to account for serial correlation in the error term for a given dyad.
-   **Logit:** Estimating non-linear models (like Logit) with many fixed effects suffers from the incidental parameters problem, which biases the coefficients. The paper uses the **Split-Panel Jackknife** (Hinz, Stammann, and Wanner, 2021) to correct this bias. Standard errors are then computed via a **cluster bootstrap of the entire jackknife procedure**, a computationally intensive but highly robust inference method.

## Features

The provided iPython Notebook (`trade_political_distance_wto_draft.ipynb`) implements the full research pipeline, including:

-   **Modular, Multi-Phase Architecture:** The entire pipeline is broken down into 21 distinct, modular tasks, each with its own orchestrator function.
-   **Configuration-Driven Design:** All methodological and computational parameters are managed in an external `config.yaml` file, allowing for easy customization without code changes.
-   **Advanced GDELT Data Processing:** A complete Particle Filter implementation to generate a smoothed, high-quality measure of political relations from raw event data.
-   **State-of-the-Art Estimation:** High-performance estimation of PPML and Logit models with multiple high-dimensional fixed effects using the `pyfixest` library.
-   **Rigorous Inference:**
    -   Complete implementation of the Split-Panel Jackknife for bias correction.
    -   A parallelized cluster bootstrap of the jackknife procedure for valid standard errors.
    -   A full implementation of the multivariate Delta Method for calculating the standard errors of non-linear marginal effects.
-   **Comprehensive Analysis Suite:** Functions to quantify institutional effects, test for temporal stability, and perform formal hypothesis tests (Wald, F-tests).
-   **Parallelized Robustness Framework:** A powerful, extensible orchestrator for running dozens of robustness checks in parallel using `joblib`.
-   **Automated Reporting:** Programmatic generation of publication-quality tables (`modelsummary`), figures (`seaborn`), and a final JSON synthesis report.

## Methodology Implemented

The core analytical steps directly implement the methodology from the paper:

1.  **Validation & Preprocessing (Tasks 1-6):** Ingests and rigorously validates all raw data and the `config.yaml` file, performs a deep data quality audit, runs the Particle Filter to create the GDELT measure, and constructs all other analytical variables.
2.  **Model & Sample Preparation (Tasks 7-9):** Adds fixed effects identifiers and interaction terms, creates the specific analytical samples (including domestic trade), and prepares generators for bootstrap and jackknife inference.
3.  **Estimation (Tasks 10-11):** Estimates the baseline PPML and Logit models using the advanced methods described above.
4.  **Analysis & Interpretation (Tasks 12-15):** Runs model diagnostics, calculates economically meaningful marginal effects, quantifies the key institutional effects, and analyzes their stability over time.
5.  **Robustness & Reporting (Tasks 16-21):** Orchestrates the entire suite of robustness checks and generates all final tables, figures, and a synthesis report.

## Core Components (Notebook Structure)

The `trade_political_distance_wto_draft.ipynb` notebook is structured as a logical pipeline with modular orchestrator functions for each of the major tasks. All functions are self-contained, fully documented with type hints and docstrings, and designed for professional-grade execution.

## Key Callable: execute_full_research_pipeline

The central function in this project is `execute_full_research_pipeline`. It orchestrates the entire analytical workflow, providing a single entry point for running the baseline study and all associated robustness checks.

```python
def execute_full_research_pipeline(
    master_df: pd.DataFrame,
    config: Dict[str, Any],
    output_dir: str,
    run_intensive_logit: bool = True,
    run_robustness_checks: bool = True
) -> None:
    """
    Executes the complete end-to-end research pipeline for the study.
    """
    # ... (implementation is in the notebook)
```

## Prerequisites

-   Python 3.9+
-   Core dependencies: `pandas`, `numpy`, `scipy`, `statsmodels`, `pyyaml`, `tqdm`, `joblib`, `matplotlib`, `seaborn`, `pydantic`, `pyfixest`, `modelsummary`.

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/trade_political_distance_wto.git
    cd trade_political_distance_wto
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

## Input Data Structure

The pipeline requires a single `pandas.DataFrame` and a `config.yaml` file. A mock data generation function is provided in the main notebook to create a valid example for testing. The `master_df` must conform to the 26-column schema defined in the `define_full_schema()` function, which includes dyadic trade data, country-level controls for both exporter and importer, and UNGA ideal points.

## Usage

The `trade_political_distance_wto_draft.ipynb` notebook provides a complete, step-by-step guide. The core workflow is:

1.  **Prepare Inputs:** Load your master `pandas.DataFrame`. Ensure the `config.yaml` file is present in the same directory.
2.  **Execute Pipeline:** Call the grand orchestrator function.

    ```python
    # This single call runs the entire project.
    execute_full_research_pipeline(
        master_df=my_master_data,
        config=my_config_dict,
        output_dir="./research_outputs",
        run_intensive_logit=False,       # Set to True for the full analysis
        run_robustness_checks=False      # Set to True for the full analysis
    )
    ```
3.  **Inspect Outputs:** Check the specified output directory (`./research_outputs`) for all generated tables, figures, and the final JSON report.

## Output Structure

The pipeline does not return an object. Its side effect is the creation of a structured output directory:

```
research_outputs/
│
├── table_1_main_results.csv              # Main PPML and Logit regression results
├── table_2_robustness_summary.csv        # Summary of key coefficients from robustness checks
├── figure_1_ppml_marginal_effects.png    # Plot of main marginal effects
├── figure_2_temporal_heterogeneity.png   # Plot showing evolution of effects over time
├── figure_3_robustness_stability.png     # Coefficient stability plot
└── final_synthesis_report.json           # Programmatically generated summary of all findings
```

## Project Structure

```
trade_political_distance_wto/
│
├── trade_political_distance_wto_draft.ipynb   # Main implementation notebook
├── config.yaml                                # Master configuration file
├── requirements.txt                           # Python package dependencies
├── LICENSE                                    # MIT license file
└── README.md                                  # This documentation file
```

## Customization

The pipeline is highly customizable via the `config.yaml` file. Users can easily modify all methodological parameters, such as particle filter settings, estimation choices, and robustness check definitions, without altering the core Python code.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. Adherence to PEP 8, type hinting, and comprehensive docstrings is required.

## Recommended Extensions

Future extensions could include:
-   **Alternative Estimators:** Adding support for other non-linear estimators like Negative Binomial for the intensive margin.
-   **Dynamic Panel Models:** Incorporating lagged dependent variables to model persistence in trade flows.
-   **Machine Learning Benchmarks:** Comparing the structural model's performance against predictive models like Gradient Boosting or Random Forests for forecasting trade flows under geopolitical stress.
-   **Interactive Visualization:** Building a dashboard (e.g., with Dash or Streamlit) to allow users to interactively explore the results and run custom scenarios.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this code or the methodology in your research, please cite the original paper:

```bibtex
@article{hardwick2025trade,
  title={{Trade, Political Distance and the World Trade Organization}},
  author={Hardwick, Samuel},
  journal={arXiv preprint arXiv:2509.17303},
  year={2025}
}
```

For the implementation itself, you may cite this repository:
```
Chirinda, C. (2025). A Quantitative Framework for Modeling Political Distance and Trade.
GitHub repository: https://github.com/chirindaopensource/trade_political_distance_wto
```

## Acknowledgments

-   Credit to **Samuel Hardwick** for the foundational research that forms the entire basis for this computational replication.
-   This project is built upon the exceptional tools provided by the open-source community. Sincere thanks to the developers of the scientific Python ecosystem, including **Pandas, NumPy, SciPy, Statsmodels, Pydantic, Joblib, Matplotlib, Seaborn, PyFixest, and Modelsummary**, whose work makes complex computational analysis accessible and robust.

--

*This README was generated based on the structure and content of `trade_political_distance_wto_draft.ipynb` and follows best practices for research software documentation.*
