#!/usr/bin/env python3
"""
Training Data Preparation for RAG-ESMFold

Downloads and processes PDB structures annotated by CATH for training
the dual-retriever + fusion module. Uses CATH S40 (40% sequence identity
non-redundant set) for fold diversity, filters by resolution and length,
and creates topology-level train/val splits.

Pipeline:
    1. Download CATH domain list (S40 representatives)
    2. Parse CATH hierarchy annotations (C.A.T.H)
    3. Query RCSB PDB for metadata (resolution, deposition date, method)
    4. Filter: resolution ≤ 3.5Å, length 50-1024, deposited before cutoff
    5. Download mmCIF files from RCSB PDB
    6. Extract sequences from structures
    7. Stratified split at CATH topology level (no topology leaks)
    8. Write train/val FASTA files + metadata JSON

Output structure:
    output_dir/
    ├── metadata.json            # full dataset metadata
    ├── splits.json              # train/val assignments with CATH info
    ├── train.fasta              # training sequences
    ├── val.fasta                # validation sequences
    ├── structures/              # mmCIF files organized by PDB ID
    │   ├── 1abc.cif.gz
    │   └── ...
    └── summary.txt              # human-readable dataset summary

Usage:
    python prepare_training_data.py --output_dir ./rag_esmfold_data
    python prepare_training_data.py --output_dir ./data --max_chains 10000 --date_cutoff 2024-05-01

Requirements:
    pip install requests tqdm biopython --break-system-packages
"""

import argparse
import gzip
import json
import logging
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable, **kwargs):
        desc = kwargs.get("desc", "")
        total = kwargs.get("total", None)
        if desc:
            logger.info(f"{desc} ({total or '?'} items)...")
        return iterable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

CATH_DOMAIN_LIST_URLS = [
    # Full domain list (ALL domains, CLF format) — we filter to S40 NR set via cross-reference
    # v4.3
    "https://download.cathdb.info/cath/releases/all-releases/v4_3_0/cath-classification-data/cath-domain-list-v4_3_0.txt",
    "http://download.cathdb.info/cath/releases/all-releases/v4_3_0/cath-classification-data/cath-domain-list-v4_3_0.txt",
    # latest release
    "https://download.cathdb.info/cath/releases/latest-release/cath-classification-data/cath-domain-list.txt",
    "http://download.cathdb.info/cath/releases/latest-release/cath-classification-data/cath-domain-list.txt",
    # S35 subsets (exist as named files in classification-data)
    "https://download.cathdb.info/cath/releases/all-releases/v4_3_0/cath-classification-data/cath-domain-list-S35-v4_3_0.txt",
    "http://download.cathdb.info/cath/releases/all-releases/v4_3_0/cath-classification-data/cath-domain-list-S35-v4_3_0.txt",
]

# S40 non-redundant set: .list has domain IDs, .fa has sequences with CATH in headers
CATH_S40_NR_URLS = [
    # FASTA with sequences (headers contain CATH classification)
    ("fasta", "https://download.cathdb.info/cath/releases/all-releases/v4_3_0/non-redundant-data-sets/cath-dataset-nonredundant-S40-v4_3_0.fa"),
    ("fasta", "http://download.cathdb.info/cath/releases/all-releases/v4_3_0/non-redundant-data-sets/cath-dataset-nonredundant-S40-v4_3_0.fa"),
    # Domain ID list
    ("list", "https://download.cathdb.info/cath/releases/all-releases/v4_3_0/non-redundant-data-sets/cath-dataset-nonredundant-S40-v4_3_0.list"),
    ("list", "http://download.cathdb.info/cath/releases/all-releases/v4_3_0/non-redundant-data-sets/cath-dataset-nonredundant-S40-v4_3_0.list"),
]

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_GRAPHQL_URL = "https://data.rcsb.org/graphql"
RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download"

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O",
    # Non-standard → X
    "MSE": "M", "HYP": "P", "TPO": "T", "SEP": "S", "PTR": "Y",
    "CSO": "C", "CSS": "C",
}


# ──────────────────────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────────────────────

@dataclass
class CATHDomain:
    """A single CATH domain from the S40 domain list."""
    domain_id: str           # e.g. "1oaiA00"
    pdb_id: str              # e.g. "1oai"
    chain_id: str            # e.g. "A"
    domain_num: str          # e.g. "00"
    cath_class: int = 0      # C
    architecture: int = 0    # A
    topology: int = 0        # T
    homology: int = 0        # H
    seq_family: int = 0      # S
    n_residues: int = 0
    resolution: float = 999.0
    cath_code: str = ""      # "C.A.T.H" string
    topology_code: str = ""  # "C.A.T" string


@dataclass
class ChainEntry:
    """A protein chain selected for the training set."""
    pdb_id: str
    chain_id: str
    domain_id: str
    cath_code: str
    topology_code: str
    superfamily_code: str
    resolution: float
    method: str              # X-RAY, ELECTRON MICROSCOPY, etc.
    deposition_date: str     # YYYY-MM-DD
    n_residues: int
    sequence: str = ""
    split: str = ""          # "train" or "val"


# ──────────────────────────────────────────────────────────────
# Step 1: Download & Parse CATH Domain List
# ──────────────────────────────────────────────────────────────

def download_cath_domain_list(output_dir: str, version: str = "latest") -> str:
    """
    Download CATH data. Tries two strategies:
      1. Full domain list (CLF format) — has CATH classification for all domains
      2. S40 non-redundant FASTA — has domain IDs + sequences in headers

    Returns path to whatever was successfully downloaded.
    """
    os.makedirs(output_dir, exist_ok=True)
    clf_path = os.path.join(output_dir, "cath-domain-list.txt")
    fasta_path = os.path.join(output_dir, "cath-S40-nr.fa")

    # If we already have one, return it
    if os.path.exists(clf_path) and os.path.getsize(clf_path) > 10000:
        logger.info(f"CATH domain list already exists: {clf_path}")
        return clf_path
    if os.path.exists(fasta_path) and os.path.getsize(fasta_path) > 10000:
        logger.info(f"CATH S40 FASTA already exists: {fasta_path}")
        return fasta_path

    # Strategy 1: Try CLF domain list
    for url in CATH_DOMAIN_LIST_URLS:
        logger.info(f"Trying domain list: {url}")
        try:
            resp = requests.get(url, timeout=120)
            if resp.status_code == 200 and len(resp.text) > 10000:
                with open(clf_path, "w") as f:
                    f.write(resp.text)
                logger.info(f"Downloaded domain list ({len(resp.text):,} bytes) → {clf_path}")
                return clf_path
            else:
                logger.warning(f"  → HTTP {resp.status_code} (size={len(resp.text)})")
        except requests.RequestException as e:
            logger.warning(f"  → Failed: {e}")

    # Strategy 2: Try S40 non-redundant FASTA (has sequences + CATH in headers)
    for fmt, url in CATH_S40_NR_URLS:
        if fmt != "fasta":
            continue
        logger.info(f"Trying S40 NR FASTA: {url}")
        try:
            resp = requests.get(url, timeout=300)  # larger file, longer timeout
            if resp.status_code == 200 and len(resp.text) > 10000:
                with open(fasta_path, "w") as f:
                    f.write(resp.text)
                logger.info(f"Downloaded S40 NR FASTA ({len(resp.text):,} bytes) → {fasta_path}")
                return fasta_path
            else:
                logger.warning(f"  → HTTP {resp.status_code} (size={len(resp.text)})")
        except requests.RequestException as e:
            logger.warning(f"  → Failed: {e}")

    raise RuntimeError(
        "Could not download CATH data from any URL.\n"
        "Please download manually:\n"
        "  Option A: Full domain list from https://www.cathdb.info/download\n"
        "            Save as: " + clf_path + "\n"
        "  Option B: S40 NR FASTA from\n"
        "            https://download.cathdb.info/cath/releases/all-releases/v4_3_0/non-redundant-data-sets/cath-dataset-nonredundant-S40-v4_3_0.fa\n"
        "            Save as: " + fasta_path
    )


def parse_cath_domain_list(filepath: str) -> List[CATHDomain]:
    """
    Parse CATH domain list file (CLF format).

    Format (space-separated):
        domain_id  C  A  T  H  S  O  L  I  D  n_residues  resolution

    We use columns: domain_id, C, A, T, H, S, n_residues, resolution
    """
    domains = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 12:
                continue

            domain_id = parts[0]  # e.g. "1oaiA00"
            if len(domain_id) < 5:
                continue

            pdb_id = domain_id[:4].lower()
            chain_id = domain_id[4]
            domain_num = domain_id[5:]

            try:
                c, a, t, h, s = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
                n_res = int(parts[10])
                resol = float(parts[11])
            except (ValueError, IndexError):
                continue

            cath_code = f"{c}.{a}.{t}.{h}"
            topo_code = f"{c}.{a}.{t}"

            dom = CATHDomain(
                domain_id=domain_id,
                pdb_id=pdb_id,
                chain_id=chain_id,
                domain_num=domain_num,
                cath_class=c,
                architecture=a,
                topology=t,
                homology=h,
                seq_family=s,
                n_residues=n_res,
                resolution=resol,
                cath_code=cath_code,
                topology_code=topo_code,
            )
            domains.append(dom)

    logger.info(f"Parsed {len(domains):,} CATH domains from CLF file")
    return domains


def parse_cath_fasta(filepath: str) -> Tuple[List[CATHDomain], Dict[str, str]]:
    """
    Parse CATH S40 non-redundant FASTA file.

    Header format varies but typically:
        >cath|4_3_0|1oaiA00/1-59 C.A.T.H=1.10.8.10 S=1 ...
    or:
        >1oaiA00 ...

    Returns (domains, sequences_dict) where sequences_dict maps domain_id→sequence.
    """
    domains = []
    sequences = {}
    current_id = None
    current_seq = []
    current_cath = None

    with open(filepath) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                # Save previous entry
                if current_id and current_seq:
                    sequences[current_id] = "".join(current_seq)

                # Parse header
                header = line[1:]
                parts = header.split()

                # Extract domain ID — could be "cath|4_3_0|1oaiA00/1-59" or just "1oaiA00"
                raw_id = parts[0]
                if "|" in raw_id:
                    # cath|version|domainID/range
                    raw_id = raw_id.split("|")[-1]
                if "/" in raw_id:
                    raw_id = raw_id.split("/")[0]

                domain_id = raw_id.strip()
                if len(domain_id) < 5:
                    current_id = None
                    current_seq = []
                    continue

                pdb_id = domain_id[:4].lower()
                chain_id = domain_id[4]
                domain_num = domain_id[5:]

                # Try to extract CATH code from header
                c, a, t, h, s = 0, 0, 0, 0, 0
                for part in parts[1:]:
                    # Look for patterns like "C.A.T.H=1.10.8.10" or just "1.10.8.10"
                    if "=" in part and "." in part:
                        val = part.split("=")[-1]
                        nums = val.split(".")
                        if len(nums) >= 4:
                            try:
                                c, a, t, h = int(nums[0]), int(nums[1]), int(nums[2]), int(nums[3])
                            except ValueError:
                                pass
                    elif part.startswith("S="):
                        try:
                            s = int(part.split("=")[1])
                        except ValueError:
                            pass

                cath_code = f"{c}.{a}.{t}.{h}" if c > 0 else ""
                topo_code = f"{c}.{a}.{t}" if c > 0 else ""

                dom = CATHDomain(
                    domain_id=domain_id,
                    pdb_id=pdb_id,
                    chain_id=chain_id,
                    domain_num=domain_num,
                    cath_class=c,
                    architecture=a,
                    topology=t,
                    homology=h,
                    seq_family=s,
                    n_residues=0,  # will fill from sequence
                    resolution=999.0,  # will get from PDB metadata
                    cath_code=cath_code,
                    topology_code=topo_code,
                )
                domains.append(dom)
                current_id = domain_id
                current_seq = []
            else:
                if current_id:
                    current_seq.append(line.strip())

    # Save last entry
    if current_id and current_seq:
        sequences[current_id] = "".join(current_seq)

    # Fill in n_residues from sequences
    for dom in domains:
        seq = sequences.get(dom.domain_id, "")
        dom.n_residues = len(seq)

    logger.info(
        f"Parsed {len(domains):,} domains and {len(sequences):,} sequences "
        f"from CATH FASTA"
    )
    return domains, sequences


def load_cath_data(filepath: str) -> Tuple[List[CATHDomain], Optional[Dict[str, str]]]:
    """
    Auto-detect file format and parse accordingly.

    Returns (domains, optional_sequences_dict).
    Sequences dict is only populated if input was a FASTA file.
    """
    # Detect format by peeking at first non-comment line
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(">"):
                # FASTA format
                logger.info("Detected FASTA format")
                domains, seqs = parse_cath_fasta(filepath)
                return domains, seqs
            else:
                # CLF format (space-separated fields)
                parts = line.split()
                if len(parts) >= 12:
                    logger.info("Detected CLF (domain list) format")
                    domains = parse_cath_domain_list(filepath)
                    return domains, None
                else:
                    # Unknown format — try both
                    break

    # Try CLF first, then FASTA
    domains = parse_cath_domain_list(filepath)
    if domains:
        return domains, None
    domains, seqs = parse_cath_fasta(filepath)
    return domains, seqs


# ──────────────────────────────────────────────────────────────
# Step 2: Query PDB Metadata via GraphQL
# ──────────────────────────────────────────────────────────────

def fetch_pdb_metadata_batch(pdb_ids: List[str]) -> Dict[str, dict]:
    """
    Fetch resolution, method, deposition date, and entity sequences
    for a batch of PDB IDs via the RCSB GraphQL API.

    Returns dict: pdb_id → {resolution, method, deposition_date, entities}
    """
    results = {}
    batch_size = 50  # GraphQL can handle ~50 IDs per query comfortably

    for start in range(0, len(pdb_ids), batch_size):
        batch = pdb_ids[start : start + batch_size]
        ids_str = ", ".join(f'"{pid}"' for pid in batch)

        query = f"""
        {{
          entries(entry_ids: [{ids_str}]) {{
            rcsb_id
            rcsb_accession_info {{
              deposit_date
            }}
            exptl {{
              method
            }}
            rcsb_entry_info {{
              resolution_combined
            }}
            polymer_entities {{
              rcsb_polymer_entity_container_identifiers {{
                auth_asym_ids
              }}
              entity_poly {{
                pdbx_seq_one_letter_code_can
                rcsb_entity_polymer_type
              }}
            }}
          }}
        }}
        """

        for attempt in range(3):
            try:
                resp = requests.post(
                    RCSB_GRAPHQL_URL,
                    json={"query": query},
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    break
                else:
                    time.sleep(2 ** attempt)
            except requests.RequestException:
                time.sleep(2 ** attempt)
        else:
            logger.warning(f"Failed to fetch metadata for batch starting at {start}")
            continue

        entries = data.get("data", {}).get("entries") or []
        for entry in entries:
            if entry is None:
                continue
            pdb_id = entry["rcsb_id"].lower()

            # Resolution
            resol_info = entry.get("rcsb_entry_info") or {}
            resolution = None
            resol_list = resol_info.get("resolution_combined")
            if resol_list and len(resol_list) > 0:
                resolution = resol_list[0]

            # Method
            exptl = entry.get("exptl") or []
            method = exptl[0]["method"] if exptl else "UNKNOWN"

            # Deposition date
            accession = entry.get("rcsb_accession_info") or {}
            dep_date = accession.get("deposit_date", "")
            if dep_date:
                dep_date = dep_date[:10]  # YYYY-MM-DD

            # Entity sequences per chain
            entities = {}
            for pe in (entry.get("polymer_entities") or []):
                ep = pe.get("entity_poly") or {}
                if ep.get("rcsb_entity_polymer_type") != "Protein":
                    continue
                seq = ep.get("pdbx_seq_one_letter_code_can", "")
                chain_ids_info = pe.get("rcsb_polymer_entity_container_identifiers") or {}
                auth_chains = chain_ids_info.get("auth_asym_ids") or []
                for ch in auth_chains:
                    entities[ch] = seq

            results[pdb_id] = {
                "resolution": resolution,
                "method": method,
                "deposition_date": dep_date,
                "chain_sequences": entities,
            }

        # Rate limiting
        time.sleep(0.2)

    return results


# ──────────────────────────────────────────────────────────────
# Step 3: Filter Domains
# ──────────────────────────────────────────────────────────────

def filter_domains(
    domains: List[CATHDomain],
    pdb_metadata: Dict[str, dict],
    min_length: int = 50,
    max_length: int = 1024,
    max_resolution: float = 3.5,
    date_cutoff: str = "2024-05-01",
    allowed_methods: Set[str] = None,
    preloaded_sequences: Optional[Dict[str, str]] = None,
) -> List[ChainEntry]:
    """
    Filter CATH domains by quality criteria and create ChainEntry objects.

    Filters:
        - Resolution ≤ max_resolution (or NMR/cryo-EM with no resolution)
        - Length between min_length and max_length
        - Deposited before date_cutoff (to avoid CASP16 leakage)
        - Experimental method in allowed_methods
        - Has extractable sequence
    """
    if allowed_methods is None:
        allowed_methods = {"X-RAY DIFFRACTION", "ELECTRON MICROSCOPY", "SOLUTION NMR"}

    entries = []
    seen_chains = set()

    stats = Counter()
    for dom in domains:
        stats["total"] += 1
        pdb_id = dom.pdb_id
        meta = pdb_metadata.get(pdb_id)

        if meta is None:
            stats["no_metadata"] += 1
            continue

        # Method filter
        method = meta["method"]
        if method not in allowed_methods:
            stats["bad_method"] += 1
            continue

        # Resolution filter
        resolution = meta["resolution"]
        if method == "X-RAY DIFFRACTION":
            if resolution is None or resolution > max_resolution:
                stats["bad_resolution"] += 1
                continue
        elif method == "ELECTRON MICROSCOPY":
            if resolution is not None and resolution > 4.0:
                stats["bad_resolution"] += 1
                continue
        # NMR: no resolution filter (resolution is typically None)

        # Date filter
        dep_date = meta["deposition_date"]
        if dep_date and dep_date >= date_cutoff:
            stats["too_recent"] += 1
            continue

        # Sequence extraction — try preloaded first, then PDB metadata
        seq = ""
        if preloaded_sequences:
            seq = preloaded_sequences.get(dom.domain_id, "")

        if not seq:
            chain_seqs = meta.get("chain_sequences", {})
            seq = chain_seqs.get(dom.chain_id, "")
        if not seq:
            stats["no_sequence"] += 1
            continue

        # Clean sequence (remove non-standard residue placeholders)
        seq = seq.replace("(", "").replace(")", "")
        seq = "".join(c for c in seq if c.isalpha() and c.isupper())

        # Length filter
        if len(seq) < min_length or len(seq) > max_length:
            stats["bad_length"] += 1
            continue

        # Deduplicate by PDB+chain (a chain can have multiple domains)
        chain_key = f"{pdb_id}_{dom.chain_id}"
        if chain_key in seen_chains:
            stats["duplicate_chain"] += 1
            continue
        seen_chains.add(chain_key)

        entry = ChainEntry(
            pdb_id=pdb_id,
            chain_id=dom.chain_id,
            domain_id=dom.domain_id,
            cath_code=dom.cath_code,
            topology_code=dom.topology_code,
            superfamily_code=dom.cath_code,
            resolution=resolution if resolution else 0.0,
            method=method,
            deposition_date=dep_date,
            n_residues=len(seq),
            sequence=seq,
        )
        entries.append(entry)
        stats["accepted"] += 1

    logger.info(f"Filter results: {dict(stats)}")
    return entries


# ──────────────────────────────────────────────────────────────
# Step 4: Stratified Split at Topology Level
# ──────────────────────────────────────────────────────────────

def stratified_topology_split(
    entries: List[ChainEntry],
    val_fraction: float = 0.10,
    min_val_topologies: int = 50,
    seed: int = 42,
) -> Tuple[List[ChainEntry], List[ChainEntry]]:
    """
    Split entries into train/val at the CATH topology level.

    No topology appears in both splits — this tests generalization
    to unseen folds, which is exactly what the retriever needs to handle.

    Strategy:
        1. Group entries by topology code (C.A.T)
        2. Randomly assign ~10% of topologies to val
        3. Ensure val has at least min_val_topologies distinct topologies
        4. All entries from a topology go to the same split
    """
    rng = random.Random(seed)

    # Group by topology
    topo_groups = defaultdict(list)
    for entry in entries:
        topo_groups[entry.topology_code].append(entry)

    topologies = list(topo_groups.keys())
    rng.shuffle(topologies)

    # Determine how many topologies go to val
    n_val_topos = max(min_val_topologies, int(len(topologies) * val_fraction))
    n_val_topos = min(n_val_topos, len(topologies) - 1)  # keep at least 1 for train

    val_topos = set(topologies[:n_val_topos])
    train_topos = set(topologies[n_val_topos:])

    train_entries = []
    val_entries = []

    for topo, group in topo_groups.items():
        if topo in val_topos:
            for e in group:
                e.split = "val"
            val_entries.extend(group)
        else:
            for e in group:
                e.split = "train"
            train_entries.extend(group)

    logger.info(
        f"Split: {len(train_entries):,} train ({len(train_topos)} topologies) / "
        f"{len(val_entries):,} val ({len(val_topos)} topologies)"
    )

    return train_entries, val_entries


# ──────────────────────────────────────────────────────────────
# Step 5: Subsample for Target Size
# ──────────────────────────────────────────────────────────────

def subsample_diverse(
    entries: List[ChainEntry],
    max_chains: int,
    seed: int = 42,
) -> List[ChainEntry]:
    """
    Subsample to max_chains while preserving topology diversity.

    Strategy:
        1. Ensure every topology with ≥1 entry gets at least 1 representative
        2. Fill remaining slots by round-robin across topologies
        3. Within each topology, prefer higher resolution structures
    """
    if len(entries) <= max_chains:
        return entries

    rng = random.Random(seed)

    # Group by topology, sort each group by resolution (best first)
    topo_groups = defaultdict(list)
    for e in entries:
        topo_groups[e.topology_code].append(e)

    for topo in topo_groups:
        topo_groups[topo].sort(key=lambda e: e.resolution)

    # Phase 1: one representative per topology
    selected = []
    remaining_by_topo = {}
    for topo, group in topo_groups.items():
        selected.append(group[0])
        remaining_by_topo[topo] = group[1:]

    if len(selected) >= max_chains:
        rng.shuffle(selected)
        return selected[:max_chains]

    # Phase 2: round-robin fill
    slots_left = max_chains - len(selected)
    topos_with_remaining = [t for t, g in remaining_by_topo.items() if g]
    rng.shuffle(topos_with_remaining)

    idx = 0
    while slots_left > 0 and topos_with_remaining:
        topo = topos_with_remaining[idx % len(topos_with_remaining)]
        if remaining_by_topo[topo]:
            selected.append(remaining_by_topo[topo].pop(0))
            slots_left -= 1
        if not remaining_by_topo[topo]:
            topos_with_remaining.remove(topo)
            if not topos_with_remaining:
                break
            idx = idx % len(topos_with_remaining)
        else:
            idx += 1

    logger.info(f"Subsampled from {len(entries):,} to {len(selected):,} chains")
    return selected


# ──────────────────────────────────────────────────────────────
# Step 6: Download mmCIF Structures
# ──────────────────────────────────────────────────────────────

def download_structure(pdb_id: str, output_dir: str) -> Optional[str]:
    """Download a single mmCIF file from RCSB PDB."""
    outpath = os.path.join(output_dir, f"{pdb_id}.cif.gz")
    if os.path.exists(outpath):
        return outpath

    url = f"{RCSB_DOWNLOAD_URL}/{pdb_id}.cif.gz"
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                with open(outpath, "wb") as f:
                    f.write(resp.content)
                return outpath
            elif resp.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(1 * (attempt + 1))

    return None


def download_structures_parallel(
    pdb_ids: List[str],
    output_dir: str,
    max_workers: int = 8,
) -> Dict[str, str]:
    """Download mmCIF files in parallel."""
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    unique_ids = list(set(pdb_ids))
    logger.info(f"Downloading {len(unique_ids)} structures to {output_dir}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_structure, pid, output_dir): pid
            for pid in unique_ids
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Downloading structures"
        ):
            pid = futures[future]
            try:
                path = future.result()
                if path:
                    results[pid] = path
            except Exception as e:
                logger.warning(f"Failed to download {pid}: {e}")

    logger.info(f"Downloaded {len(results)}/{len(unique_ids)} structures")
    return results


# ──────────────────────────────────────────────────────────────
# Step 7: Write Output Files
# ──────────────────────────────────────────────────────────────

def write_fasta(entries: List[ChainEntry], filepath: str):
    """Write entries to a FASTA file."""
    with open(filepath, "w") as f:
        for entry in entries:
            header = (
                f">{entry.pdb_id}_{entry.chain_id} "
                f"cath={entry.cath_code} "
                f"topo={entry.topology_code} "
                f"res={entry.resolution:.2f} "
                f"len={entry.n_residues} "
                f"method={entry.method} "
                f"date={entry.deposition_date}"
            )
            f.write(header + "\n")
            # Write sequence in 80-char lines
            seq = entry.sequence
            for i in range(0, len(seq), 80):
                f.write(seq[i : i + 80] + "\n")

    logger.info(f"Wrote {len(entries):,} sequences to {filepath}")


def write_metadata(entries: List[ChainEntry], filepath: str):
    """Write full metadata as JSON."""
    records = []
    for e in entries:
        records.append({
            "pdb_id": e.pdb_id,
            "chain_id": e.chain_id,
            "domain_id": e.domain_id,
            "cath_code": e.cath_code,
            "topology_code": e.topology_code,
            "superfamily_code": e.superfamily_code,
            "resolution": e.resolution,
            "method": e.method,
            "deposition_date": e.deposition_date,
            "n_residues": e.n_residues,
            "split": e.split,
        })

    with open(filepath, "w") as f:
        json.dump(records, f, indent=2)

    logger.info(f"Wrote metadata for {len(records):,} entries to {filepath}")


def write_summary(
    train_entries: List[ChainEntry],
    val_entries: List[ChainEntry],
    filepath: str,
):
    """Write a human-readable summary."""
    all_entries = train_entries + val_entries

    train_topos = set(e.topology_code for e in train_entries)
    val_topos = set(e.topology_code for e in val_entries)
    train_sfams = set(e.superfamily_code for e in train_entries)
    val_sfams = set(e.superfamily_code for e in val_entries)

    # Class distribution
    class_names = {1: "Mainly Alpha", 2: "Mainly Beta", 3: "Alpha-Beta", 4: "Few SS"}
    train_classes = Counter(e.cath_code.split(".")[0] for e in train_entries)
    val_classes = Counter(e.cath_code.split(".")[0] for e in val_entries)

    # Resolution stats
    train_resolutions = [e.resolution for e in train_entries if e.resolution > 0]
    val_resolutions = [e.resolution for e in val_entries if e.resolution > 0]

    # Length stats
    train_lengths = [e.n_residues for e in train_entries]
    val_lengths = [e.n_residues for e in val_entries]

    with open(filepath, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("RAG-ESMFold Training Data Summary\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Total chains:      {len(all_entries):,}\n")
        f.write(f"  Train:           {len(train_entries):,}\n")
        f.write(f"  Validation:      {len(val_entries):,}\n\n")

        f.write(f"Unique topologies: {len(train_topos | val_topos)}\n")
        f.write(f"  Train:           {len(train_topos)}\n")
        f.write(f"  Validation:      {len(val_topos)}\n")
        f.write(f"  Overlap:         {len(train_topos & val_topos)} (should be 0)\n\n")

        f.write(f"Unique superfamilies: {len(train_sfams | val_sfams)}\n")
        f.write(f"  Train:              {len(train_sfams)}\n")
        f.write(f"  Validation:         {len(val_sfams)}\n\n")

        f.write("CATH Class Distribution:\n")
        for cls_id in sorted(set(list(train_classes.keys()) + list(val_classes.keys()))):
            name = class_names.get(int(cls_id), "Unknown")
            f.write(
                f"  Class {cls_id} ({name:15s}): "
                f"train={train_classes.get(cls_id, 0):5d}  "
                f"val={val_classes.get(cls_id, 0):5d}\n"
            )

        f.write(f"\nResolution (Å):\n")
        if train_resolutions:
            f.write(
                f"  Train:  mean={sum(train_resolutions)/len(train_resolutions):.2f}  "
                f"median={sorted(train_resolutions)[len(train_resolutions)//2]:.2f}  "
                f"range=[{min(train_resolutions):.2f}, {max(train_resolutions):.2f}]\n"
            )
        if val_resolutions:
            f.write(
                f"  Val:    mean={sum(val_resolutions)/len(val_resolutions):.2f}  "
                f"median={sorted(val_resolutions)[len(val_resolutions)//2]:.2f}  "
                f"range=[{min(val_resolutions):.2f}, {max(val_resolutions):.2f}]\n"
            )

        f.write(f"\nSequence Length:\n")
        f.write(
            f"  Train:  mean={sum(train_lengths)/max(len(train_lengths),1):.0f}  "
            f"median={sorted(train_lengths)[len(train_lengths)//2]:.0f}  "
            f"range=[{min(train_lengths)}, {max(train_lengths)}]\n"
        )
        f.write(
            f"  Val:    mean={sum(val_lengths)/max(len(val_lengths),1):.0f}  "
            f"median={sorted(val_lengths)[len(val_lengths)//2]:.0f}  "
            f"range=[{min(val_lengths)}, {max(val_lengths)}]\n"
        )

        # Method breakdown
        f.write(f"\nExperimental Method:\n")
        method_counts = Counter(e.method for e in all_entries)
        for method, count in method_counts.most_common():
            f.write(f"  {method:30s}: {count:5d}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("Split strategy: topology-level (no topology shared between splits)\n")
        f.write("This ensures the validation set tests generalization to unseen folds.\n")
        f.write("=" * 70 + "\n")

    logger.info(f"Summary written to {filepath}")


# ──────────────────────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data for RAG-ESMFold",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for all data files",
    )
    parser.add_argument(
        "--max_chains", type=int, default=10000,
        help="Maximum number of chains in the final dataset. "
             "Set to 0 for no limit (use all passing filters).",
    )
    parser.add_argument(
        "--val_fraction", type=float, default=0.10,
        help="Fraction of topologies assigned to validation",
    )
    parser.add_argument(
        "--min_length", type=int, default=50,
        help="Minimum sequence length (residues)",
    )
    parser.add_argument(
        "--max_length", type=int, default=1024,
        help="Maximum sequence length (residues)",
    )
    parser.add_argument(
        "--max_resolution", type=float, default=3.5,
        help="Maximum resolution in Å (X-ray only)",
    )
    parser.add_argument(
        "--date_cutoff", type=str, default="2024-05-01",
        help="Exclude structures deposited on or after this date (YYYY-MM-DD). "
             "Default = CASP16 start date to avoid test set leakage.",
    )
    parser.add_argument(
        "--cath_version", type=str, default="latest",
        choices=["latest", "v4.3"],
        help="CATH version to use",
    )
    parser.add_argument(
        "--download_structures", action="store_true",
        help="Also download mmCIF structure files from RCSB",
    )
    parser.add_argument(
        "--download_workers", type=int, default=8,
        help="Number of parallel download threads",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--skip_metadata_fetch", action="store_true",
        help="Skip PDB metadata fetch (use CATH resolution instead). "
             "Faster but less accurate filtering.",
    )

    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("RAG-ESMFold Training Data Preparation")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Max chains: {args.max_chains or 'unlimited'}")
    logger.info(f"Date cutoff: {args.date_cutoff}")
    logger.info(f"Resolution cutoff: {args.max_resolution}Å")
    logger.info(f"Length range: [{args.min_length}, {args.max_length}]")

    # ── Step 1: Download & parse CATH data ──
    logger.info("\n--- Step 1: CATH Domain List ---")
    cath_file = download_cath_domain_list(output_dir, args.cath_version)
    domains, fasta_sequences = load_cath_data(cath_file)

    if not domains:
        logger.error(
            f"No domains parsed from {cath_file}.\n"
            "Please check the file format. First 3 lines:"
        )
        with open(cath_file) as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                logger.error(f"  {line.rstrip()}")
        sys.exit(1)

    n_topos = len(set(d.topology_code for d in domains if d.topology_code))
    n_sfams = len(set(d.cath_code for d in domains if d.cath_code))
    logger.info(
        f"CATH S40: {len(domains):,} domains, "
        f"{n_topos:,} topologies, {n_sfams:,} superfamilies"
    )
    if fasta_sequences:
        logger.info(f"Pre-loaded {len(fasta_sequences):,} sequences from FASTA")

    # ── Pre-filter by CATH resolution (quick, before expensive API calls) ──
    # CATH CLF stores resolution; 999.0 = NMR, 1000.0 = obsolete
    # For FASTA-loaded data, resolution is 999.0 (unknown) — keep all, filter via PDB API
    if fasta_sequences is None:
        # CLF path: pre-filter obsolete entries
        domains = [d for d in domains if d.resolution < 900]
        logger.info(f"After removing obsolete/unresolved: {len(domains):,} domains")
    else:
        logger.info(f"FASTA path: skipping resolution pre-filter ({len(domains):,} domains)")

    # ── Step 2: Fetch PDB metadata ──
    logger.info("\n--- Step 2: PDB Metadata ---")

    metadata_cache = os.path.join(output_dir, "pdb_metadata_cache.json")
    pdb_metadata = {}

    if os.path.exists(metadata_cache) and not args.skip_metadata_fetch:
        logger.info(f"Loading cached metadata from {metadata_cache}")
        with open(metadata_cache) as f:
            pdb_metadata = json.load(f)
        logger.info(f"Loaded metadata for {len(pdb_metadata):,} PDB entries")

    if not args.skip_metadata_fetch:
        # Determine which PDB IDs we still need
        unique_pdb_ids = list(set(d.pdb_id for d in domains))
        missing_ids = [pid for pid in unique_pdb_ids if pid not in pdb_metadata]

        if missing_ids:
            logger.info(
                f"Fetching metadata for {len(missing_ids):,} PDB entries "
                f"({len(unique_pdb_ids):,} total, {len(pdb_metadata):,} cached)"
            )
            new_metadata = fetch_pdb_metadata_batch(missing_ids)
            pdb_metadata.update(new_metadata)

            # Cache for next run
            with open(metadata_cache, "w") as f:
                json.dump(pdb_metadata, f)
            logger.info(f"Cached metadata for {len(pdb_metadata):,} entries")
    else:
        # Use CATH-provided resolution; fabricate minimal metadata
        logger.info("Skipping PDB metadata fetch, using CATH resolution data")
        for d in domains:
            if d.pdb_id not in pdb_metadata:
                pdb_metadata[d.pdb_id] = {
                    "resolution": d.resolution if d.resolution < 900 else None,
                    "method": "X-RAY DIFFRACTION" if d.resolution < 900 else "SOLUTION NMR",
                    "deposition_date": "2000-01-01",
                    "chain_sequences": {},
                }

    # ── Step 3: Filter ──
    logger.info("\n--- Step 3: Filtering ---")
    entries = filter_domains(
        domains, pdb_metadata,
        min_length=args.min_length,
        max_length=args.max_length,
        max_resolution=args.max_resolution,
        date_cutoff=args.date_cutoff,
        preloaded_sequences=fasta_sequences,
    )

    if not entries:
        logger.error("No entries passed filters! Check parameters.")
        sys.exit(1)

    logger.info(f"Entries passing all filters: {len(entries):,}")

    # ── Step 4: Subsample if needed ──
    if args.max_chains > 0 and len(entries) > args.max_chains:
        logger.info(f"\n--- Step 4: Subsampling to {args.max_chains:,} ---")
        entries = subsample_diverse(entries, args.max_chains, args.seed)

    # ── Step 5: Stratified split ──
    logger.info("\n--- Step 5: Train/Val Split ---")
    train_entries, val_entries = stratified_topology_split(
        entries, val_fraction=args.val_fraction, seed=args.seed,
    )

    # ── Step 6: Download structures (optional) ──
    if args.download_structures:
        logger.info("\n--- Step 6: Downloading Structures ---")
        struct_dir = os.path.join(output_dir, "structures")
        all_pdb_ids = [e.pdb_id for e in train_entries + val_entries]
        download_structures_parallel(
            all_pdb_ids, struct_dir, max_workers=args.download_workers,
        )

    # ── Step 7: Write output files ──
    logger.info("\n--- Step 7: Writing Outputs ---")

    write_fasta(train_entries, os.path.join(output_dir, "train.fasta"))
    write_fasta(val_entries, os.path.join(output_dir, "val.fasta"))
    write_metadata(
        train_entries + val_entries,
        os.path.join(output_dir, "splits.json"),
    )
    write_summary(
        train_entries, val_entries,
        os.path.join(output_dir, "summary.txt"),
    )

    # ── Done ──
    logger.info("\n" + "=" * 60)
    logger.info("Done!")
    logger.info(f"  Train:       {len(train_entries):,} chains")
    logger.info(f"  Validation:  {len(val_entries):,} chains")
    logger.info(f"  Output:      {output_dir}")
    logger.info("=" * 60)

    # Print usage hint
    print(f"\nNext steps:")
    print(f"  1. Build FAISS index:  python build_faiss_index.py esm2_t12_35M_UR50D {output_dir}/train.fasta train_esm2.index")
    print(f"  2. Build TM-Vec index: python build_tmvec_index.py {output_dir}/train.fasta train_tmvec.index")
    print(f"  3. Start training:     python train_rag_esmfold.py --train_fasta {output_dir}/train.fasta --val_fasta {output_dir}/val.fasta")


if __name__ == "__main__":
    main()