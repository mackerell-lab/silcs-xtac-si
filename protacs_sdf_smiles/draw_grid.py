#!/usr/bin/env python
import svgutils.transform as sg
import sys
import os

def assemble_svg_grid(svg_paths, ncols, scale=1.0, label_fontsize=14):
  labels = [os.path.splitext(os.path.basename(p))[0] for p in svg_paths]
  nrows = (len(svg_paths) + ncols - 1) // ncols

  cell_width, cell_height = 500, 500
  fig_width = f"{ncols * cell_width}px"
  fig_height = f"{nrows * (cell_height + label_fontsize + 0)}px"

  fig = sg.SVGFigure(fig_width, fig_height)

  for i, (path, label) in enumerate(zip(svg_paths, labels)):
    col = i % ncols
    row = i // ncols
    x = col * cell_width
    y = row * cell_height

    mol = sg.fromfile(path).getroot()
    mol.moveto(x, y)
    mol.scale(scale)

    txt = sg.TextElement(x, y+label_fontsize, label, size=label_fontsize, weight='bold')
    fig.append([mol, txt])

  return fig

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} mol1.svg mol2.svg ...")
    sys.exit(1)

  svg_paths = sys.argv[1:]
  final_svg = assemble_svg_grid(svg_paths, ncols=4, label_fontsize=30)
  final_svg.save("combined_grid.svg")

