# Data Details

## Overview
This folder provides all the real-world data for the COVID-19 experiments and the necessary scripts to generate the synthetic datasets from the SIR model from Kermack et al. (1927).

## File Structure
- The raw real-world data for the experiments in contained in /real-world/raw.
- The selected subset of the this real-world dataset is contained in /real-world/covid
- /real-world/sum_cases_over_counties.py aggregates the infection numbers of the raw dataset over the counties to get a global view (German perspective) of the infection dynamic.
- In plot.ipynb, we create all the plots used to visualize the paper
- Running sir.py generates the synthetic datasets from the fitted SIR model.
- We use calculate_new_cases.py to transfer the generated cumulative cases from the SIR model to new infections.
- We provide an exemplary synthetic datasets in /synthetic (already transformed)

## Data Sources
The real-world data with the infection waves from the COVID-19 pandemic stems from Robert Koch-Institut (https://doi.org/10.5281/zenodo.7042186)

## Data Preprocessing
Details on pre-processing steps are described in the appendix of the paper.
