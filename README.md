# silcs-protac-si
Supporting Information for SILCS-PROTAC method paper, Nordquist et al.

## Contents:
* protac-loocv.py: python script to perform LOO-CV/RFE fitting and produce figures to describe fitting
* pair_list_dc50.txt : file describing list of data to loop over (enables looping over a subset as desired)
* protac_data.csv: csv file containing SILCS-PROTAC metrics and experimental DC50 data
* protac_data.pkl: python pkl file containing SILCS-PROTAC metrics and experimental DC50 data
* environment.yml: conda environment yml file to replicate the python packages used by silcs-protac

## Directories:
* loocv_rfe_results: csv of LOO-CV/RFE results
* figs: contains LOO-CV/RFE figures and general models aggregated for each number of SILCS-PROTAC metric subsets
* random-trial: results from 10 replicas of LOO-CV/RFE fitting on randomized data
* protacs_sdf: directories for each system, contains sdf files for protacs, target warheads and ligase warheads

