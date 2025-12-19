#!/usr/bin/env python3
import nbformat
import sys

def find_incomplete_assignments(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            print(f"\n=== CELL {i} ===")
            for j, line in enumerate(cell.source.split('\n'), 1):
                # Check for lines with assignment but no value after equals
                stripped = line.strip()
                if '=' in stripped and not stripped.endswith('\\'):
                    # Find position of last '=' in the line
                    parts = stripped.split('=')
                    if len(parts) > 1 and not parts[-1].strip():
                        # There's an assignment but nothing after the last '='
                        print(f"Line {j}: {line!r}")
                        print(f"      ^ POTENTIAL INCOMPLETE ASSIGNMENT")

find_incomplete_assignments('/workspace/Практическая 4 (Seafood) Титов Павел УСБО-02-22 .ipynb')