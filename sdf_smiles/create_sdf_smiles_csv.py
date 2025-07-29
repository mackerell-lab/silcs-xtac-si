#!/home/enord/local/miniconda3/envs/silcsprotac/bin/python
from rdkit import Chem
import sys,os

smiles_csv = open('smiles.csv', 'w')
smiles_csv.write('System,Compound_ID,PROTAC,Ligase_Warhead,Target_Warhead\n')

for directory in os.listdir('./'):
  # Load the molecules from input files
  if os.path.isdir(directory):
    protacs   = Chem.SDMolSupplier(directory+'/protacs.sdf', sanitize=True, removeHs=True)
    warheads1 = Chem.SDMolSupplier(directory+'/aligned_washed_ligase_warheads.sdf', sanitize=True, removeHs=True)
    warheads2 = Chem.SDMolSupplier(directory+'/aligned_washed_target_warheads.sdf', sanitize=True, removeHs=True)
    
    for protac, w1, w2 in zip(protacs, warheads1, warheads2):
      name = protac.GetProp('_Name')
      protac_smiles = Chem.MolToSmiles(protac)
      w1_smiles = Chem.MolToSmiles(w1)
      w2_smiles = Chem.MolToSmiles(w2)
      smiles_csv.write(f"{directory},{name},{protac_smiles},{w1_smiles},{w2_smiles}\n")
