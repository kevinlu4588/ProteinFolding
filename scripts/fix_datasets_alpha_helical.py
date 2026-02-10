"""
Extract all-alpha protein sequences directly from CATH Class 1 (Mainly Alpha).

This script:
1. Downloads CATH domain list
2. Filters for Class 1 (Mainly Alpha)
3. Downloads structures and extracts sequences
4. Optionally verifies zero beta content with DSSP
5. Saves full chain sequences

CATH non-redundant sets:
- S100: 100% sequence identity (all domains)
- S95, S60, S40, S35: progressively less redundant
"""

import os
import urllib.request
from collections import defaultdict
from Bio.PDB import PDBList, MMCIFParser
from Bio.PDB.DSSP import DSSP

# ============================================================
# CONFIGURATION
# ============================================================
OUTPUT_FASTA = "cath_all_alpha.fasta"
OUTPUT_CSV = "cath_all_alpha_nohup.csv"
CIF_DIR = "./cif_files"
CATH_FILE = "cath-domain-list.txt"

# Filters
MIN_LENGTH = 50
MAX_LENGTH = 400
VERIFY_WITH_DSSP = True  # Set False to skip DSSP verification (faster but less strict)
MAX_PROTEINS = None  # Stop after collecting this many proteins

os.makedirs(CIF_DIR, exist_ok=True)

# ============================================================
# STEP 1: Download and parse CATH
# ============================================================
def download_cath():
    """Download CATH domain list."""
    url = "http://download.cathdb.info/cath/releases/latest-release/cath-classification-data/cath-domain-list.txt"
    
    if os.path.exists(CATH_FILE):
        print(f"Using existing: {CATH_FILE}")
        return
    
    print(f"Downloading CATH domain list...")
    urllib.request.urlretrieve(url, CATH_FILE)
    print(f"Saved to {CATH_FILE}")


def parse_cath_class1():
    """
    Parse CATH and return Class 1 (Mainly Alpha) domains.
    
    File format (CLF 2.0):
    Col 1: domain_id (e.g., 1oaiA00)
    Col 2: Class (1=alpha, 2=beta, 3=alpha-beta, 4=few SS)
    Col 3: Architecture
    Col 4: Topology
    Col 5: Homologous superfamily
    Col 6-9: S35, S60, S95, S100 cluster IDs
    Col 10: Domain length
    Col 11: Resolution (999.000 = NMR)
    
    Returns dict: {(pdb_id, chain): [domain_info, ...]}
    """
    domains = defaultdict(list)
    
    with open(CATH_FILE, 'r') as f:
        for line in f:
            # Skip comments and empty lines
            if line.startswith('#') or not line.strip():
                continue
            
            parts = line.strip().split()
            if len(parts) < 11:
                continue
            
            domain_id = parts[0]      # e.g., "1oaiA00"
            cath_class = parts[1]     # 1, 2, 3, or 4
            cath_arch = parts[2]      # Architecture
            cath_topol = parts[3]     # Topology  
            cath_homol = parts[4]     # Homologous superfamily
            domain_length = parts[10] # Length
            resolution = parts[11] if len(parts) > 11 else None
            
            # Class 1 = Mainly Alpha
            if cath_class != '1':
                continue
            
            pdb_id = domain_id[:4].lower()
            chain = domain_id[4]
            domain_num = domain_id[5:]
            
            # Build CATH code string
            cath_code = f"{cath_class}.{cath_arch}.{cath_topol}.{cath_homol}"
            
            try:
                length = int(domain_length)
            except:
                length = None
            
            try:
                res = float(resolution) if resolution else None
            except:
                res = None
            
            domains[(pdb_id, chain)].append({
                'domain_id': domain_id,
                'cath_code': cath_code,
                'domain_num': domain_num,
                'length': length,
                'resolution': res,
            })
    
    return domains


# ============================================================
# STEP 2: Extract sequences with optional DSSP verification
# ============================================================
def extract_sequence(pdb_id, chain, cif_dir, pdbl, parser, verify_dssp=True):
    """
    Download structure and extract sequence for a chain.
    Optionally verify zero beta content with DSSP.
    
    Returns: (sequence, helix_frac, error_msg)
    """
    # Download
    try:
        file_path = pdbl.retrieve_pdb_file(pdb_id, pdir=cif_dir, file_format='mmCif')
    except Exception as e:
        return None, None, f"Download failed: {e}"
    
    # Parse
    try:
        structure = parser.get_structure(pdb_id, file_path)
        model = structure[0]
    except Exception as e:
        return None, None, f"Parse failed: {e}"
    
    if chain not in model:
        return None, None, f"Chain {chain} not found"
    
    if verify_dssp:
        # Run DSSP
        try:
            dssp = DSSP(model, file_path, dssp='mkdssp')
        except Exception as e:
            return None, None, f"DSSP failed: {e}"
        
        # Extract residues for target chain
        chain_residues = []
        for key in dssp.keys():
            if key[0] == chain:
                res_num = key[1][1]
                aa = dssp[key][1]
                ss = dssp[key][2]
                chain_residues.append((res_num, aa, ss))
        
        if not chain_residues:
            return None, None, "No DSSP residues"
        
        # Check for beta
        beta_codes = {'E', 'B'}
        n_beta = sum(1 for _, _, ss in chain_residues if ss in beta_codes)
        
        if n_beta > 0:
            return None, None, f"Has {n_beta} beta residues"
        
        # Extract sequence
        chain_residues.sort(key=lambda x: x[0])
        sequence = ''.join(aa for _, aa, _ in chain_residues if aa != 'X')
        
        # Helix fraction
        helix_codes = {'H', 'G', 'I'}
        n_helix = sum(1 for _, _, ss in chain_residues if ss in helix_codes)
        helix_frac = n_helix / len(chain_residues) if chain_residues else 0
        
    else:
        # Just extract sequence without DSSP
        from Bio.PDB.Polypeptide import protein_letters_3to1
        
        residues = []
        for res in model[chain]:
            if res.id[0] == ' ':  # Standard residue
                resname = res.resname
                if resname in protein_letters_3to1:
                    residues.append((res.id[1], protein_letters_3to1[resname]))
        
        if not residues:
            return None, None, "No residues found"
        
        residues.sort(key=lambda x: x[0])
        sequence = ''.join(aa for _, aa in residues)
        helix_frac = None  # Unknown without DSSP
    
    return sequence, helix_frac, None


# ============================================================
# STEP 3: Save results
# ============================================================
def save_results(results, fasta_path, csv_path):
    """Save results to FASTA and CSV files."""
    with open(fasta_path, 'w') as f:
        for p in results:
            hf = f"|helix={p['helix_fraction']:.2f}" if p['helix_fraction'] else ""
            f.write(f">{p['pdb_id']}{p['chain']}|len={p['length']}{hf}|{p['cath_codes']}\n")
            f.write(f"{p['sequence']}\n")
    
    with open(csv_path, 'w') as f:
        f.write("pdb_id,chain,length,helix_fraction,cath_codes,sequence\n")
        for p in results:
            hf = f"{p['helix_fraction']:.3f}" if p['helix_fraction'] else ""
            f.write(f"{p['pdb_id']},{p['chain']},{p['length']},{hf},{p['cath_codes']},{p['sequence']}\n")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("CATH Class 1 (Mainly Alpha) Protein Extraction")
    print("=" * 60)
    print(f"  Length filter: {MIN_LENGTH}-{MAX_LENGTH} residues")
    print(f"  DSSP verification: {VERIFY_WITH_DSSP}")
    print(f"  Max proteins: {MAX_PROTEINS or 'unlimited'}")
    
    # Step 1: Get CATH Class 1 domains
    print("\n[1] Loading CATH Class 1 domains...")
    download_cath()
    cath_domains = parse_cath_class1()
    print(f"    Found {len(cath_domains)} unique (pdb, chain) pairs in Class 1")
    
    # Step 2: Process each chain
    print(f"\n[2] Extracting sequences...")
    pdbl = PDBList()
    parser = MMCIFParser(QUIET=True)
    
    results = []
    seen_sequences = set()
    downloaded_pdbs = {}  # Cache file paths
    
    chains_list = list(cath_domains.keys())
    
    for idx, (pdb_id, chain) in enumerate(chains_list):
        # Check if we've hit the limit
        if MAX_PROTEINS and len(results) >= MAX_PROTEINS:
            print(f"\n    Reached limit of {MAX_PROTEINS} proteins")
            break
        
        # Progress
        if (idx + 1) % 100 == 0:
            print(f"    Progress: {idx+1}/{len(chains_list)}, found {len(results)} all-alpha")
        
        # Extract sequence
        seq, helix_frac, error = extract_sequence(
            pdb_id, chain, CIF_DIR, pdbl, parser, 
            verify_dssp=VERIFY_WITH_DSSP
        )
        
        if error:
            # Show rejections due to beta content
            if "beta" in error.lower():
                print(f"    [skip] {pdb_id}{chain}: {error}")
            continue
        
        # Length filter
        if not (MIN_LENGTH <= len(seq) <= MAX_LENGTH):
            print(f"    [skip] {pdb_id}{chain}: length {len(seq)} outside {MIN_LENGTH}-{MAX_LENGTH}")
            continue
        
        # Skip duplicates
        if seq in seen_sequences:
            continue
        seen_sequences.add(seq)
        
        # Get CATH info for this chain
        domain_info = cath_domains[(pdb_id, chain)]
        cath_codes = ','.join(set(d['cath_code'] for d in domain_info))
        
        results.append({
            'pdb_id': pdb_id,
            'chain': chain,
            'sequence': seq,
            'length': len(seq),
            'helix_fraction': helix_frac,
            'cath_codes': cath_codes,
        })
        
        hf_str = f"{helix_frac:.0%}" if helix_frac else "N/A"
        print(f"    [{len(results)}/{MAX_PROTEINS}] {pdb_id}{chain}: len={len(seq)}, helix={hf_str}")
        
        # Flush results every 10 proteins
        if len(results) % 10 == 0:
            save_results(results, OUTPUT_FASTA, OUTPUT_CSV)
            print(f"    ** Saved checkpoint: {len(results)} proteins **")
    
    # Step 3: Save
    print(f"\n[3] Saving {len(results)} proteins...")
    save_results(results, OUTPUT_FASTA, OUTPUT_CSV)
    
    # Summary
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  Total CATH Class 1 chains: {len(cath_domains)}")
    print(f"  Passed all filters: {len(results)}")
    print(f"  Output: {OUTPUT_FASTA}")
    print(f"  Output: {OUTPUT_CSV}")
    
    if results:
        lengths = [p['length'] for p in results]
        print(f"\n  Length range: {min(lengths)}-{max(lengths)}")
        if results[0]['helix_fraction'] is not None:
            hfs = [p['helix_fraction'] for p in results if p['helix_fraction']]
            print(f"  Helix fraction: {min(hfs):.0%}-{max(hfs):.0%}")


if __name__ == "__main__":
    main()