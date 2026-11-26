#!/usr/bin/env python3
"""
Build a human-only PPI network for CAFA6.

This script:
1. Parses CAFA train/test sequences to get UniProt IDs
2. Maps STRING IDs to UniProt using the alias file
3. Builds PPI edges from STRING (with score threshold)
4. Builds PPI edges from BioGRID (using gene symbol → UniProt mapping)
5. Merges both sources into a unified PPI network
6. Creates a neighbor index for fast lookups

Usage:
    python build_human_ppi.py [--min_score 700]
"""

import argparse
import os
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import pandas as pd  # type: ignore

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = "."  # All raw files are here
OUT_DIR = "data_ppi"  # Output directory

# File names (adjust if needed)
TRAIN_FASTA = "train_sequences.fasta"
TEST_FASTA = "testsuperset.fasta"
STRING_ALIASES = "9606.protein.aliases.v12.0.txt"
STRING_LINKS = "9606.protein.links.detailed.v12.0 14.17.52.txt"  # Note: has space in name
BIOGRID_FILE = "BIOGRID-ORGANISM-Homo_sapiens-5.0.252.tab2.txt"

# Default parameters
DEFAULT_MIN_SCORE = 700  # STRING combined_score threshold (0-1000)
BIOGRID_DEFAULT_WEIGHT = 0.9  # Default weight for BioGRID interactions
CHUNK_SIZE = 500_000  # Chunk size for reading large files


# =============================================================================
# Helper Functions
# =============================================================================

def normalize_protein_id(raw_id: str) -> str:
    """
    Extract UniProt accession from FASTA header.
    
    Examples:
        ">sp|P9WHI7|RECN_MYCT ..." → "P9WHI7"
        ">Q9CQV8|10090" → "Q9CQV8"
        ">A0A0C5B5G6 9606" → "A0A0C5B5G6"
    """
    raw_id = raw_id.lstrip(">").strip()
    
    # Handle sp|ACCESSION|NAME format
    if raw_id.startswith("sp|") or raw_id.startswith("tr|"):
        parts = raw_id.split("|")
        if len(parts) >= 2:
            return parts[1].split()[0]
    
    # Handle ACCESSION|TAXON or ACCESSION TAXON format
    if "|" in raw_id:
        return raw_id.split("|")[0].strip()
    
    # Just take first part before whitespace
    return raw_id.split()[0].strip()


def is_valid_uniprot_id(protein_id: str) -> bool:
    """
    Check if a string looks like a valid UniProt accession.
    UniProt accessions are 6-10 alphanumeric characters.
    Pattern: [OPQ][0-9][A-Z0-9]{3}[0-9] or [A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}
    """
    if not protein_id or len(protein_id) < 6 or len(protein_id) > 10:
        return False
    # Simple pattern: starts with letter, alphanumeric
    return bool(re.match(r'^[A-Z][A-Z0-9]{5,9}$', protein_id))


def sorted_pair(a: str, b: str) -> Tuple[str, str]:
    """Return a sorted tuple of two protein IDs (for edge deduplication)."""
    return (a, b) if a <= b else (b, a)


# =============================================================================
# Step 1: Load CAFA Protein IDs
# =============================================================================

def load_cafa_ids(data_dir: str) -> Set[str]:
    """
    Parse CAFA train and test FASTA files to extract all unique UniProt accessions.
    
    Returns:
        Set of normalized UniProt accession IDs
    """
    cafa_ids = set()
    
    fasta_files = [
        os.path.join(data_dir, TRAIN_FASTA),
        os.path.join(data_dir, TEST_FASTA),
    ]
    
    for fasta_path in fasta_files:
        if not os.path.exists(fasta_path):
            print(f"  [WARNING] FASTA file not found: {fasta_path}")
            continue
            
        print(f"  Reading: {fasta_path}")
        with open(fasta_path, "r") as f:
            for line in f:
                if line.startswith(">"):
                    protein_id = normalize_protein_id(line)
                    if protein_id:
                        cafa_ids.add(protein_id)
    
    return cafa_ids


# =============================================================================
# Step 2: Build STRING ID → UniProt Mapping
# =============================================================================

def build_string_mapping(data_dir: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Parse STRING alias file to build:
    1. STRING protein ID → UniProt accession mapping
    2. Gene symbol → UniProt accession mapping (for BioGRID)
    
    Returns:
        Tuple of (string_to_uniprot, gene_to_uniprot) dicts
    """
    alias_path = os.path.join(data_dir, STRING_ALIASES)
    
    if not os.path.exists(alias_path):
        raise FileNotFoundError(f"STRING alias file not found: {alias_path}")
    
    print(f"  Reading: {alias_path}")
    
    # Store all UniProt accessions per STRING ID
    string_to_uniprots: Dict[str, list] = defaultdict(list)
    # Store gene symbol → UniProt mapping
    gene_to_uniprot: Dict[str, str] = {}
    
    # Also build reverse mapping: UniProt → STRING ID (for gene symbol resolution)
    uniprot_to_string: Dict[str, str] = {}
    
    # Read in chunks to handle large file
    for chunk in pd.read_csv(
        alias_path,
        sep="\t",
        header=0,
        names=["string_protein_id", "alias", "source"],
        dtype=str,
        chunksize=CHUNK_SIZE,
        comment="#",
        on_bad_lines="skip",
    ):
        # Filter for UniProt_AC source (primary UniProt accessions)
        uniprot_rows = chunk[chunk["source"] == "UniProt_AC"]
        
        for _, row in uniprot_rows.iterrows():
            string_id = row["string_protein_id"]
            alias = row["alias"]
            if pd.notna(string_id) and pd.notna(alias):
                string_to_uniprots[string_id].append(alias)
                uniprot_to_string[alias] = string_id
        
        # Also collect gene names for BioGRID mapping
        gene_rows = chunk[chunk["source"] == "UniProt_GN_Name"]
        for _, row in gene_rows.iterrows():
            string_id = row["string_protein_id"]
            gene_symbol = row["alias"]
            if pd.notna(string_id) and pd.notna(gene_symbol):
                # Get UniProt ID for this STRING ID
                if string_id in string_to_uniprots and string_to_uniprots[string_id]:
                    # Map gene symbol to first UniProt ID
                    gene_to_uniprot[gene_symbol.upper()] = string_to_uniprots[string_id][0]
    
    # Finalize: pick preferred UniProt ID for each STRING ID
    string_to_uniprot = {}
    for string_id, uniprots in string_to_uniprots.items():
        if uniprots:
            # Prefer the first accession, or the shortest (canonical)
            string_to_uniprot[string_id] = min(uniprots, key=len)
    
    # Second pass: complete gene → UniProt mapping for genes we missed
    print(f"  Second pass for gene symbol mapping...")
    for chunk in pd.read_csv(
        alias_path,
        sep="\t",
        header=0,
        names=["string_protein_id", "alias", "source"],
        dtype=str,
        chunksize=CHUNK_SIZE,
        comment="#",
        on_bad_lines="skip",
    ):
        gene_rows = chunk[chunk["source"] == "UniProt_GN_Name"]
        for _, row in gene_rows.iterrows():
            string_id = row["string_protein_id"]
            gene_symbol = row["alias"]
            if pd.notna(string_id) and pd.notna(gene_symbol):
                gene_upper = gene_symbol.upper()
                if gene_upper not in gene_to_uniprot and string_id in string_to_uniprot:
                    gene_to_uniprot[gene_upper] = string_to_uniprot[string_id]
    
    return string_to_uniprot, gene_to_uniprot


# =============================================================================
# Step 3: Build STRING PPI Edges
# =============================================================================

def build_string_ppi(
    data_dir: str,
    string_to_uniprot: Dict[str, str],
    cafa_ids: Set[str],
    min_score: int = DEFAULT_MIN_SCORE,
) -> pd.DataFrame:
    """
    Parse STRING interactions and filter to CAFA proteins.
    
    Args:
        data_dir: Directory containing STRING files
        string_to_uniprot: Mapping from STRING ID to UniProt
        cafa_ids: Set of CAFA UniProt IDs
        min_score: Minimum combined_score (0-1000)
    
    Returns:
        DataFrame with columns: prot_a, prot_b, weight, source
    """
    links_path = os.path.join(data_dir, STRING_LINKS)
    
    if not os.path.exists(links_path):
        raise FileNotFoundError(f"STRING links file not found: {links_path}")
    
    print(f"  Reading: {links_path}")
    print(f"  Using score threshold: {min_score}")
    
    edges: Dict[Tuple[str, str], float] = {}
    
    n_rows = 0
    n_mapped = 0
    n_filtered = 0
    
    for chunk in pd.read_csv(
        links_path,
        sep=r"\s+",  # Space-separated
        chunksize=CHUNK_SIZE,
        dtype={"protein1": str, "protein2": str, "combined_score": int},
        usecols=["protein1", "protein2", "combined_score"],
    ):
        n_rows += len(chunk)
        
        # Filter by score threshold
        chunk = chunk[chunk["combined_score"] >= min_score]
        n_filtered += len(chunk)
        
        for _, row in chunk.iterrows():
            p1 = row["protein1"]
            p2 = row["protein2"]
            score = row["combined_score"]
            
            # Map to UniProt
            u1 = string_to_uniprot.get(p1)
            u2 = string_to_uniprot.get(p2)
            
            # Skip if either cannot be mapped
            if not u1 or not u2:
                continue
            
            # Skip if not in CAFA IDs
            if u1 not in cafa_ids or u2 not in cafa_ids:
                continue
            
            # Skip self-loops
            if u1 == u2:
                continue
            
            n_mapped += 1
            
            # Create sorted edge and compute weight
            edge = sorted_pair(u1, u2)
            weight = score / 1000.0
            
            # Keep max weight for duplicate edges
            if edge not in edges or edges[edge] < weight:
                edges[edge] = weight
    
    print(f"    Total rows read: {n_rows:,}")
    print(f"    After score filter: {n_filtered:,}")
    print(f"    Mapped to CAFA: {n_mapped:,}")
    print(f"    Unique edges: {len(edges):,}")
    
    # Build DataFrame
    df = pd.DataFrame([
        {"prot_a": e[0], "prot_b": e[1], "weight": w, "source": "STRING"}
        for e, w in edges.items()
    ])
    
    return df


# =============================================================================
# Step 4: Build BioGRID PPI Edges
# =============================================================================

def build_biogrid_ppi(
    data_dir: str,
    gene_to_uniprot: Dict[str, str],
    cafa_ids: Set[str],
    default_weight: float = BIOGRID_DEFAULT_WEIGHT,
) -> pd.DataFrame:
    """
    Parse BioGRID interactions and filter to CAFA proteins.
    
    Uses gene symbol → UniProt mapping from STRING aliases.
    
    Args:
        data_dir: Directory containing BioGRID file
        gene_to_uniprot: Mapping from gene symbol to UniProt
        cafa_ids: Set of CAFA UniProt IDs
        default_weight: Default edge weight
    
    Returns:
        DataFrame with columns: prot_a, prot_b, weight, source
    """
    biogrid_path = os.path.join(data_dir, BIOGRID_FILE)
    
    if not os.path.exists(biogrid_path):
        raise FileNotFoundError(f"BioGRID file not found: {biogrid_path}")
    
    print(f"  Reading: {biogrid_path}")
    
    # Read BioGRID (tab-separated)
    # Only read the columns we need
    usecols = [
        "Official Symbol Interactor A",
        "Official Symbol Interactor B",
        "Organism Interactor A",
        "Organism Interactor B",
    ]
    
    edges: Dict[Tuple[str, str], float] = {}
    
    n_rows = 0
    n_human = 0
    n_mapped = 0
    
    for chunk in pd.read_csv(
        biogrid_path,
        sep="\t",
        usecols=usecols,
        dtype=str,
        chunksize=CHUNK_SIZE,
        on_bad_lines="skip",
    ):
        n_rows += len(chunk)
        
        # Filter for human-human interactions (taxon 9606)
        chunk = chunk[
            (chunk["Organism Interactor A"] == "9606") &
            (chunk["Organism Interactor B"] == "9606")
        ]
        n_human += len(chunk)
        
        for _, row in chunk.iterrows():
            gene_a = row["Official Symbol Interactor A"]
            gene_b = row["Official Symbol Interactor B"]
            
            if pd.isna(gene_a) or pd.isna(gene_b):
                continue
            
            # Map gene symbols to UniProt
            u1 = gene_to_uniprot.get(gene_a.upper())
            u2 = gene_to_uniprot.get(gene_b.upper())
            
            # Skip if either cannot be mapped
            if not u1 or not u2:
                continue
            
            # Skip if not in CAFA IDs
            if u1 not in cafa_ids or u2 not in cafa_ids:
                continue
            
            # Skip self-loops
            if u1 == u2:
                continue
            
            n_mapped += 1
            
            # Create sorted edge
            edge = sorted_pair(u1, u2)
            
            # Keep max weight for duplicates
            if edge not in edges or edges[edge] < default_weight:
                edges[edge] = default_weight
    
    print(f"    Total rows read: {n_rows:,}")
    print(f"    Human-human interactions: {n_human:,}")
    print(f"    Mapped to CAFA: {n_mapped:,}")
    print(f"    Unique edges: {len(edges):,}")
    
    # Build DataFrame
    df = pd.DataFrame([
        {"prot_a": e[0], "prot_b": e[1], "weight": w, "source": "BioGRID"}
        for e, w in edges.items()
    ])
    
    return df


# =============================================================================
# Step 5: Merge PPI Networks
# =============================================================================

def merge_ppi(
    ppi_string: pd.DataFrame,
    ppi_biogrid: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge STRING and BioGRID PPI networks.
    
    For each unique edge:
    - weight = max weight from all sources
    - sources = comma-joined list of sources
    - n_sources = count of distinct sources
    - from_string, from_biogrid = boolean flags
    
    Returns:
        Merged DataFrame
    """
    # Combine both DataFrames
    combined = pd.concat([ppi_string, ppi_biogrid], ignore_index=True)
    
    if len(combined) == 0:
        return pd.DataFrame(columns=[
            "prot_a", "prot_b", "weight", "sources", "n_sources",
            "from_string", "from_biogrid"
        ])
    
    # Group by edge
    merged_edges = []
    
    for (prot_a, prot_b), group in combined.groupby(["prot_a", "prot_b"]):
        sources = set(group["source"].tolist())
        max_weight = group["weight"].max()
        
        merged_edges.append({
            "prot_a": prot_a,
            "prot_b": prot_b,
            "weight": max_weight,
            "sources": ",".join(sorted(sources)),
            "n_sources": len(sources),
            "from_string": 1 if "STRING" in sources else 0,
            "from_biogrid": 1 if "BioGRID" in sources else 0,
        })
    
    df = pd.DataFrame(merged_edges)
    return df


def build_neighbor_index(ppi_merged: pd.DataFrame) -> Dict[str, list]:
    """
    Build a neighbor index: protein → list of (neighbor, weight).
    
    This is an undirected graph, so each edge appears twice.
    """
    neighbors: Dict[str, list] = defaultdict(list)
    
    for _, row in ppi_merged.iterrows():
        a, b = row["prot_a"], row["prot_b"]
        w = row["weight"]
        neighbors[a].append((b, w))
        neighbors[b].append((a, w))
    
    # Convert to regular dict
    return dict(neighbors)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build human PPI network for CAFA6"
    )
    parser.add_argument(
        "--min_score",
        type=int,
        default=DEFAULT_MIN_SCORE,
        help=f"Minimum STRING combined_score (0-1000). Default: {DEFAULT_MIN_SCORE}",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DATA_DIR,
        help=f"Directory with input files. Default: {DATA_DIR}",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=OUT_DIR,
        help=f"Output directory. Default: {OUT_DIR}",
    )
    args = parser.parse_args()
    
    data_dir = args.data_dir
    out_dir = args.out_dir
    min_score = args.min_score
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Step 1: Load CAFA protein IDs
    # -------------------------------------------------------------------------
    print("\n[Step 1] Loading CAFA protein IDs...")
    cafa_ids = load_cafa_ids(data_dir)
    print(f"  Total CAFA proteins: {len(cafa_ids):,}")
    
    # Save CAFA IDs
    cafa_ids_path = os.path.join(out_dir, "cafa_protein_ids.tsv")
    pd.DataFrame({"uniprot_id": sorted(cafa_ids)}).to_csv(
        cafa_ids_path, sep="\t", index=False
    )
    print(f"  Saved: {cafa_ids_path}")
    
    # -------------------------------------------------------------------------
    # Step 2: Build STRING mappings
    # -------------------------------------------------------------------------
    print("\n[Step 2] Building STRING ID → UniProt mapping...")
    string_to_uniprot, gene_to_uniprot = build_string_mapping(data_dir)
    print(f"  STRING → UniProt mappings: {len(string_to_uniprot):,}")
    print(f"  Gene → UniProt mappings: {len(gene_to_uniprot):,}")
    
    # Save STRING mapping
    string_map_path = os.path.join(out_dir, "string_to_uniprot.tsv")
    pd.DataFrame([
        {"string_id": k, "uniprot_id": v}
        for k, v in sorted(string_to_uniprot.items())
    ]).to_csv(string_map_path, sep="\t", index=False)
    print(f"  Saved: {string_map_path}")
    
    # -------------------------------------------------------------------------
    # Step 3: Build STRING PPI
    # -------------------------------------------------------------------------
    print("\n[Step 3] Building STRING PPI edges...")
    ppi_string = build_string_ppi(data_dir, string_to_uniprot, cafa_ids, min_score)
    
    # Save STRING PPI
    string_ppi_path = os.path.join(out_dir, "ppi_string.tsv")
    ppi_string.to_csv(string_ppi_path, sep="\t", index=False)
    print(f"  Saved: {string_ppi_path}")
    
    # -------------------------------------------------------------------------
    # Step 4: Build BioGRID PPI
    # -------------------------------------------------------------------------
    print("\n[Step 4] Building BioGRID PPI edges...")
    ppi_biogrid = build_biogrid_ppi(data_dir, gene_to_uniprot, cafa_ids)
    
    # Save BioGRID PPI
    biogrid_ppi_path = os.path.join(out_dir, "ppi_biogrid.tsv")
    ppi_biogrid.to_csv(biogrid_ppi_path, sep="\t", index=False)
    print(f"  Saved: {biogrid_ppi_path}")
    
    # -------------------------------------------------------------------------
    # Step 5: Merge PPI networks
    # -------------------------------------------------------------------------
    print("\n[Step 5] Merging STRING and BioGRID PPI...")
    ppi_merged = merge_ppi(ppi_string, ppi_biogrid)
    
    # Save merged PPI
    merged_ppi_path = os.path.join(out_dir, "ppi_merged.tsv")
    ppi_merged.to_csv(merged_ppi_path, sep="\t", index=False)
    print(f"  Saved: {merged_ppi_path}")
    
    # Build and save neighbor index
    print("\n[Step 6] Building neighbor index...")
    neighbor_index = build_neighbor_index(ppi_merged)
    neighbor_path = os.path.join(out_dir, "ppi_neighbors.pkl")
    with open(neighbor_path, "wb") as f:
        pickle.dump(neighbor_index, f, protocol=4)
    print(f"  Saved: {neighbor_path}")
    
    # -------------------------------------------------------------------------
    # Summary statistics
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total CAFA proteins:          {len(cafa_ids):>10,}")
    
    # Count how many CAFA proteins have at least one PPI edge
    proteins_with_ppi = set(ppi_merged["prot_a"]) | set(ppi_merged["prot_b"])
    coverage = len(proteins_with_ppi.intersection(cafa_ids))
    print(f"CAFA proteins with PPI:       {coverage:>10,} ({100*coverage/len(cafa_ids):.1f}%)")
    
    print(f"Edges from STRING:            {len(ppi_string):>10,}")
    print(f"Edges from BioGRID:           {len(ppi_biogrid):>10,}")
    print(f"Edges after merging:          {len(ppi_merged):>10,}")
    
    # Source breakdown
    if len(ppi_merged) > 0:
        n_both = len(ppi_merged[ppi_merged["n_sources"] == 2])
        n_string_only = len(ppi_merged[(ppi_merged["from_string"] == 1) & (ppi_merged["from_biogrid"] == 0)])
        n_biogrid_only = len(ppi_merged[(ppi_merged["from_string"] == 0) & (ppi_merged["from_biogrid"] == 1)])
        print(f"  - STRING only:              {n_string_only:>10,}")
        print(f"  - BioGRID only:             {n_biogrid_only:>10,}")
        print(f"  - Both sources:             {n_both:>10,}")
    
    print(f"Proteins in neighbor index:   {len(neighbor_index):>10,}")
    print("=" * 60)
    print("\nDone! Output files are in:", out_dir)


if __name__ == "__main__":
    main()

