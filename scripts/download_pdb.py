import os
import numpy as np
from Bio.PDB import PDBList, MMCIFParser
from Bio.SeqUtils import seq1

# 1. Configuration
DATA_DIR = "/insomnia001/depts/pmg/users/js6118/data/retrieval/pdb"
RAW_DIR = os.path.join(DATA_DIR, "raw_cif")
OUTPUT_FILE = os.path.join(DATA_DIR, "protein_samples.npz")

# Diverse set of 2022-2025 structures
PDB_IDS = ["7T0A", "8D60", "8S2S", "8F2X", "7U6Q", "8CVL", "8X9Y", "8B1H", "7R6G", "8P2V"]

def prepare_pipeline():
    os.makedirs(RAW_DIR, exist_ok=True)
    pdbl = PDBList()
    parser = MMCIFParser(QUIET=True)
    
    dataset = {}

    print(f"🚀 Starting pipeline for {len(PDB_IDS)} samples...")

    for pdb_id in PDB_IDS:
        # Download
        print(f"📦 Downloading {pdb_id}...")
        cif_file = pdbl.retrieve_pdb_file(pdb_id, pdir=RAW_DIR, file_format="mmCif")
        
        try:
            # Parse
            structure = parser.get_structure(pdb_id, cif_file)
            # Use only first model and first chain for the sample
            chain = structure[0].child_list[0]
            
            sequence = []
            ca_coords = []
            
            for residue in chain:
                # Filter for standard residues and valid CA atoms
                if "CA" in residue and residue.get_resname() in ['ALA','CYS','ASP','GLU','PHE','GLY','HIS','ILE','LYS','LEU','MET','ASN','PRO','GLN','ARG','SER','THR','VAL','TRP','TYR']:
                    sequence.append(seq1(residue.get_resname()))
                    ca_coords.append(residue["CA"].get_coord())
            
            # Store in dictionary
            if sequence:
                dataset[pdb_id] = {
                    "seq": "".join(sequence),
                    "coords": np.array(ca_coords).astype(np.float32)
                }
                print(f"✅ Processed {pdb_id}: {len(sequence)} residues")
        
        except Exception as e:
            print(f"❌ Error processing {pdb_id}: {e}")

    # 2. Save as NPZ
    np.savez(OUTPUT_FILE, **dataset)
    print(f"\n💾 Pipeline complete! Data saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    prepare_pipeline()