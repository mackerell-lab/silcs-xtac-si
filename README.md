# silcs-xtac-si
Supporting Information for SILCS-XTAC method paper, Nordquist et al.

## Contents:
* protac-loocv.py: python script to perform LOO-CV/RFE fitting and produce figures to describe fitting
* pair_list_dc50.txt : file describing list of data to loop over (enables looping over a subset as desired)
* protac_data.csv: csv file containing SILCS-PROTAC metrics and experimental DC50 data; NOTE: only assays with > 3 entries were considered in final analyses, and there are "duplicate" PROTACS which appear in multiple different, but related, assays
* protac_data.pkl: python pkl file containing SILCS-PROTAC metrics and experimental DC50 data
* environment.yml: conda environment yml file to replicate the python packages used by silcs-protac

## Directories:
* loocv_rfe_results: csv of LOO-CV/RFE results
* figs: contains LOO-CV/RFE figures and general models aggregated for each number of SILCS-PROTAC metric subsets
* random-trial: results from 10 replicas of LOO-CV/RFE fitting on randomized data
* sdf_smiles: contains smiles.csv and directories for each system containing sdf and svg (image) files of PROTACs and warheads
  * smiles.csv: csv file with smiles strings of each PROTAC and warheads
  * mcs_highlight_warheads.py: draws svg image of PROTACs with warheads highlighted
  * draw_grid.py: helper script to assemble svg images into a grid
  * create_sdf_smiles_csv.py: helper script to make smiles.csv
  * Each system directory contains the sdf format files and a grid-style svg image of the protacs with warheads highlighted

