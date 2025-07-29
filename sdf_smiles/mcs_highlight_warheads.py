#!/home/enord/local/miniconda3/envs/silcsprotac/bin/python
from rdkit import Chem
from rdkit.Chem import Draw, rdFMCS, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import sys

def draw_highlighted_warheads_in_protac(protac, warhead1, warhead2,color1=(0,0,1,0.5), color2=(1,0,0,0.5)):
  print(protac.GetProp('_Name'))
  # Find MCS between PROTAC and each warhead
  mcs1 = rdFMCS.FindMCS([protac, warhead1], completeRingsOnly=False, timeout=10)
  mcs2 = rdFMCS.FindMCS([protac, warhead2], completeRingsOnly=False, timeout=10)
  if not mcs1.smartsString or not mcs2.smartsString:
    print("Warning: Empty SMARTS returned from MCS")
    return
  mcs1_mol = Chem.MolFromSmarts(mcs1.smartsString)
  mcs2_mol = Chem.MolFromSmarts(mcs2.smartsString)
  if mcs1_mol is None or mcs2_mol is None:
    print("Warning: Failed to parse MCS SMARTS")
    return

  # Get matching atom indices in both molecules
  pmatch1 = protac.GetSubstructMatch(mcs1_mol)
  pmatch2 = protac.GetSubstructMatch(mcs2_mol)
  if not pmatch1 or not pmatch2:
    print("Warning: MCS match failed in PROTAC")
    return
  protac_match = pmatch1 + pmatch2  # Combined atom match set

  # Prepare lists of matching bonds
  protac_bond_matches = [bond.GetIdx() for bond in protac.GetBonds() if bond.GetBeginAtomIdx() in protac_match and bond.GetEndAtomIdx() in protac_match]
  pmatch_bd1 = [bond.GetIdx() for bond in protac.GetBonds() if bond.GetBeginAtomIdx() in pmatch1 and bond.GetEndAtomIdx() in pmatch1]

  # Prepare drawing options
  AllChem.Compute2DCoords(protac)
  protac = Draw.PrepareMolForDrawing(protac)

  # Assign each atom and bond a color
  atom_colors, bond_colors = {}, {}
  for at in protac_match:
    atom_colors[at] = color1 if at in pmatch1 else color2
  for bd in protac_bond_matches:
    bond_colors[bd] = color1 if bd in pmatch_bd1 else color2

  # Generate images for PROTAC with highlighted atoms and standalone warhead
  d = rdMolDraw2D.MolDraw2DSVG(500,500)
  #d.drawOptions().fillHighlights = False
  rdMolDraw2D.PrepareAndDrawMolecule(
    d, protac, 
    highlightAtoms=list(protac_match), 
    highlightAtomColors=atom_colors, 
    highlightBonds=list(protac_bond_matches), 
    highlightBondColors=bond_colors, 
  )

  d.FinishDrawing()
  filename=protac.GetProp('_Name')+'.svg'
  with open(filename, 'w') as f:
    f.write(d.GetDrawingText())

if __name__ == '__main__':
  if len(sys.argv) < 4:
    print(f'Usage: python {sys.argv[0]} protacs.sdf warheads1.sdf warheads2.sdf')
    exit(1)
  # Load the molecules from input files
  protacs   = Chem.SDMolSupplier(sys.argv[1], sanitize=True, removeHs=True )  # No Hs for clarity
  warheads1 = Chem.SDMolSupplier(sys.argv[2], sanitize=True, removeHs=True)
  warheads2 = Chem.SDMolSupplier(sys.argv[3], sanitize=True, removeHs=True)
  
  if not (len(protacs) == len(warheads1) == len(warheads2)): print("Warning: Number of molecules in SDF files do not match.")

  for protac, w1, w2 in zip(protacs, warheads1, warheads2):
    draw_highlighted_warheads_in_protac(protac, w1, w2)

