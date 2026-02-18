#!/usr/bin/env python
"""Extract FASTA sequences from PDB files."""

import sys
from pathlib import Path
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
import warnings


def pdb_to_fasta(pdb_dir, output, max_count=None):
    """Extract sequences from PDB files and write to FASTA."""
    pdb_files = sorted(Path(pdb_dir).iterdir())
    if max_count:
        pdb_files = pdb_files[:max_count]

    parser = PDBParser(QUIET=True)

    with open(output, 'w') as f:
        for pdb_file in pdb_files:
            if not pdb_file.is_file():
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                structure = parser.get_structure("protein", str(pdb_file))

            seq = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.id[0] == ' ':
                            try:
                                seq.append(seq1(residue.resname))
                            except KeyError:
                                continue

            if seq:
                f.write(f">{pdb_file.name}\n")
                f.write(''.join(seq) + '\n')

    print(f"Generated {output} with {len(pdb_files)} sequences")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python fasta_generator.py <pdb_dir> <output_fasta> [max_count]")
        sys.exit(1)

    pdb_dir = sys.argv[1]
    output = sys.argv[2]
    max_count = int(sys.argv[3]) if len(sys.argv) > 3 else None

    pdb_to_fasta(pdb_dir, output, max_count)
