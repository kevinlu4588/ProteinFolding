"""
Download all-alpha proteins from RCSB PDB.
Uses a curated list of known all-alpha protein families.
"""

import requests
import os
import time
import pandas as pd

# Curated list of known all-alpha proteins (helix bundles, etc.)
# Curated list of known all-alpha proteins (helix bundles, etc.)
KNOWN_ALL_ALPHA = [
    # ===========================================
    # MYOGLOBIN/HEMOGLOBIN FAMILY (Globin fold)
    # ===========================================
    "1MBN",  # Myoglobin (sperm whale)
    "1A6M",  # Myoglobin
    "1DWR",  # Myoglobin
    "2MM1",  # Myoglobin
    "1WLA",  # Myoglobin
    "1MBS",  # Myoglobin
    "1BZP",  # Myoglobin
    "1MBI",  # Myoglobin
    "2V1K",  # Neuroglobin
    
    # ===========================================
    # DE NOVO DESIGNED HELIX BUNDLES
    # ===========================================
    "1M6T",  # De novo 3-helix bundle
    "2A3D",  # De novo alpha-3D (three-helix bundle)
    "6DS9",  # GRa3D elongated three-helix bundle
    "1BDD",  # Protein A (3-helix bundle)
    "2JOF",  # Designed helical bundle
    "1LQ7",  # Villin headpiece
    "1VII",  # Villin headpiece subdomain HP35
    "2F4K",  # Villin headpiece
    "1YRF",  # De novo protein
    "2KHK",  # Designed helical protein
    "2P6J",  # Designed protein
    "1F4N",  # De novo 4-helix bundle
    "2LSE",  # De novo designed four-helix bundle (DND_4HB)
    "1COS",  # Synthetic triple-stranded alpha-helical bundle
    "1AL1",  # Alpha1 synthetic protein
    
    # ===========================================
    # HOMEODOMAIN PROTEINS (Helix-turn-helix)
    # ===========================================
    "1ENH",  # Engrailed homeodomain
    "1EQZ",  # Engrailed homeodomain
    "1HDD",  # Homeodomain
    "9ANT",  # Antennapedia homeodomain
    
    # ===========================================
    # REPRESSORS (Helix-turn-helix DNA binding)
    # ===========================================
    "1LMB",  # Lambda repressor
    "1R69",  # 434 repressor DNA-binding domain
    "2OR1",  # Lambda cro repressor
    
    # ===========================================
    # COILED COILS / LEUCINE ZIPPERS
    # ===========================================
    "2ZTA",  # GCN4 leucine zipper
    "1GCM",  # GCN4 coiled coil
    "1N0Q",  # Coiled coil
    "1UNO",  # Tropomyosin
    "1IC2",  # Serpin
    "1AQ5",  # Coiled coil
    "1FOS",  # Fos-Jun coiled coil
    "1KD8",  # Coiled coil
    
    # ===========================================
    # FOUR-HELIX BUNDLES (Classic)
    # ===========================================
    "1EC5",  # Cytochrome b562
    "1M3W",  # Cytochrome c'
    "1CGO",  # Cytochrome c'
    "1ROP",  # ROP protein
    "1BBH",  # ROP protein variant
    
    # ===========================================
    # FERRITINS (All-alpha storage proteins)
    # ===========================================
    "1FEX",  # Ferritin
    "2FHA",  # Ferritin
    "1TFE",  # Ferritin
    "1BCF",  # Ferritin
    "1AEW",  # Ferritin
    "1IER",  # Ferritin
    "1MFR",  # Ferritin
    
    # ===========================================
    # SPECTRIN / REPEAT PROTEINS
    # ===========================================
    "2SPZ",  # Spectrin repeat
    "1H5Q",  # Spectrin
    "1CUN",  # Spectrin repeat
    
    # ===========================================
    # EF-HAND CALCIUM BINDING (All-alpha)
    # ===========================================
    "1CLL",  # Calmodulin (Ca2+ bound)
    "1CFD",  # Calmodulin (apo)
    "1CFC",  # Calmodulin fragment
    "3CLN",  # Calmodulin
    "1B8C",  # Parvalbumin
    "1B8R",  # Parvalbumin
    "1B9A",  # Parvalbumin
    "4CPV",  # Carp parvalbumin
    "5PAL",  # Parvalbumin
    "1TCF",  # Troponin C
    "1TOP",  # Troponin C
    
    # ===========================================
    # SERUM ALBUMIN (All-alpha transport)
    # ===========================================
    "1E78",  # Human serum albumin
    "1N5U",  # Human serum albumin with heme
    "1AO6",  # Human serum albumin
    
    # ===========================================
    # APOLIPOPROTEINS (All-alpha)
    # ===========================================
    "1AV1",  # Apolipoprotein A-I
    "3R2P",  # Apolipoprotein A-I
    "1GW4",  # Apolipoprotein A-I fragment
    
    # ===========================================
    # CYTOKINES - FOUR-HELIX BUNDLE
    # ===========================================
    # Interleukins
    "1ITL",  # Interleukin-4 (four-helix bundle)
    "1RCB",  # Interleukin-4 crystal
    "1M47",  # Interleukin-2
    "1IRL",  # Interleukin-2
    "2IL6",  # Interleukin-6
    "1JLI",  # Interleukin-3
    "1HIK",  # Interleukin-5
    "1ILR",  # Interleukin-4 receptor complex
    
    # Growth factors/hormones
    "1HGU",  # Human growth hormone
    "1HWG",  # Human growth hormone with receptor
    "3HHR",  # Human growth hormone receptor complex
    "1BUY",  # Erythropoietin
    "1CN4",  # Erythropoietin
    "1AX8",  # Leptin
    "1CSG",  # GM-CSF (granulocyte-macrophage CSF)
    "1RHG",  # GM-CSF
    "1GNC",  # G-CSF
    "1CD9",  # G-CSF
    "1PRL",  # Prolactin
    
    # Interferons
    "1IFA",  # Interferon-beta
    "1ITF",  # Interferon alpha-2a
    "1AU1",  # Interferon alpha-2b
    "1RFB",  # Interferon-beta
    "1HIG",  # Interferon gamma
    
    # ===========================================
    # OTHER SMALL ALPHA-HELICAL PROTEINS
    # ===========================================
    "1UTG",  # Uteroglobin
    "1PRU",  # Pheromone
    "1P68",  # Albumin-binding domain
    "1PRB",  # Protein B IgG binding domain
    "1GAB",  # IgG binding domain
    "1HRC",  # Cytochrome c (mostly alpha)
    
    # ===========================================
    # TPR/HEAT/ARM REPEAT PROTEINS (All-alpha solenoids)
    # ===========================================
    "1NA3",  # TPR protein (designed)
    "1NAR",  # TPR protein
    "1K36",  # Ankyrin repeat
    "2AJA",  # Ankyrin repeat (Legionella)
    "1W63",  # HEAT repeat (PP2A subunit)
    "2IAE",  # HEAT repeat (PP2A)
    "3V6A",  # API5 (HEAT/ARM repeat)
    "2QFC",  # TPR repeat (PlcR)
    "2Z6H",  # Beta-catenin (ARM repeat)
    "1IAL",  # ARM repeat
    
    # ===========================================
    # BACTERIAL/SMALL HELICAL PROTEINS
    # ===========================================
    "1K9Q",  # Bacterial protein
    "1ORC",  # Arc repressor
    "1ARR",  # Arc repressor
    "1KRS",  # Ketosteroid isomerase (mainly alpha)
    "1LMB",  # Lambda cI repressor
    
    # ===========================================
    # HELICAL MEMBRANE PROTEINS (soluble domains)
    # ===========================================
    "1CWP",  # Coat protein
    
    # ===========================================
    # ADDITIONAL DESIGNED/ENGINEERED PROTEINS
    # ===========================================
    "1CC5",  # Cytochrome c552
    "1YCC",  # Cytochrome c
    "1YEA",  # Cytochrome c
    "256B",  # Cytochrome b5
    "1CYO",  # Cytochrome c oxidase subunit (soluble)
    
    # ===========================================
    # MISCELLANEOUS SMALL ALPHA PROTEINS
    # ===========================================
    "1POH",  # Phospholipase (alpha helical)
    "1HMV",  # Hemerythrin (all alpha)
    "2HMQ",  # Hemerythrin
    "1A7N",  # Hemerythrin
    "1I27",  # Titin (alpha helical domain)
    "1TIT",  # Titin
    "1AHO",  # Alpha helical protein
    "1WHZ",  # Alpha/beta hydrolase (mostly alpha)
    "1TEN",  # Tenascin (alpha helical)
    "1NKL",  # Natural killer cell (alpha helical)
    "1MBA",  # Myohemerythrin,
    # ===========================================
    # ADDITIONAL GLOBINS
    # ===========================================
    "1HBB",  # Hemoglobin beta chain
    "1HBA",  # Hemoglobin alpha chain
    "1GDI",  # Leghemoglobin
    "1DLY",  # Leghemoglobin
    "1CG5",  # Hemoglobin
    "1THB",  # Hemoglobin
    "1FLP",  # Hemoglobin
    "1ASH",  # Hemoglobin (Ascaris)
    "1HLM",  # Hemoglobin (sea lamprey)
    "1ECA",  # Erythrocruorin
    
    # ===========================================
    # MORE DE NOVO / DESIGNED PROTEINS
    # ===========================================
    "1QYS",  # De novo designed protein
    "1MFT",  # De novo alpha bundle
    "1U2H",  # Designed protein
    "1P9G",  # Designed alpha helical
    "1M4F",  # Designed helical protein
    "1T8J",  # De novo designed
    "2GJH",  # Designed helical bundle
    "2HYZ",  # Designed protein
    "1W4E",  # Designed protein
    "1W4H",  # Designed protein
    
    # ===========================================
    # MORE CYTOCHROMES (verified all-alpha)
    # ===========================================
    "1CPR",  # Cytochrome c peroxidase domain
    "1YNR",  # Cytochrome
    "1A56",  # Cytochrome
    "351C",  # Cytochrome c551
    "1C75",  # Cytochrome c7
    "1BBJ",  # Cytochrome c6
    "1QJ2",  # Cytochrome
    
    # ===========================================
    # ADDITIONAL FOUR-HELIX BUNDLES
    # ===========================================
    "1LPE",  # Lipid binding protein
    "1H97",  # Four-helix bundle
    "1BYI",  # Cytochrome
    "1DV0",  # Four-helix bundle
    "1B3A",  # Four-helix bundle
    "1APC",  # Apocytochrome b562
    "1M6T",  # Helix bundle
    
    # ===========================================
    # MORE COILED COILS (short, verified)
    # ===========================================
    "1NKN",  # Coiled coil
    "1COI",  # Coiled coil
    "1GK4",  # GCN4 variant
    "1ZII",  # Coiled coil
    "1S9Z",  # Coiled coil
    "1U0I",  # Coiled coil
    
    # ===========================================
    # ADDITIONAL EF-HAND / CALCIUM BINDING
    # ===========================================
    "1RFJ",  # S100 protein
    "1A03",  # S100 calcium binding
    "1MHO",  # Calbindin
    "1B1G",  # S100B
    "1QLK",  # Calmodulin-like
    "1NKE",  # Recoverin
    "1JSA",  # Frequenin
    
    # ===========================================
    # HELICAL BUNDLE SIGNALING PROTEINS  
    # ===========================================
    "1BH8",  # Bcl-xL (apoptosis)
    "1BH9",  # Bcl-2 family
    "1G5M",  # Bcl-2 family
    "1MAZ",  # Bcl-2
    
    # ===========================================
    # SAM/STERILE ALPHA MOTIF DOMAINS
    # ===========================================
    "1OQN",  # SAM domain
    "1PK1",  # SAM domain
    "1STP",  # SAM domain
    
    # ===========================================
    # HELICAL REPEAT DOMAINS (smaller)
    # ===========================================
    "1AWC",  # ARM repeat small
    "1LDK",  # HEAT repeat small domain
    
    # ===========================================
    # MISCELLANEOUS VERIFIED ALL-ALPHA
    # ===========================================
    "1R0R",  # Bacterial alpha
    "1TUK",  # Alpha helical
    "1DU2",  # Alpha protein
    "1FW4",  # Helical protein
    "1DIV",  # Helical DNA-binding
    "1E0L",  # Alpha helical
    "1YK4",  # Helical bundle
    "1JM0",  # All-alpha
    "2EZN",  # Designed alpha
    "1U84",  # Helical bundle
    "1WHI",  # Alpha bundle,
        # ===========================================
    # MORE GLOBINS / OXYGEN BINDING
    # ===========================================
    "1HDA",  # Hemoglobin (deoxy)
    "1MYT",  # Myoglobin (tuna)
    "1YMB",  # Myoglobin variant
    "1VXA",  # Truncated hemoglobin
    "1DLW",  # Hemoglobin
    "1ITH",  # Hemoglobin
    "1SCT",  # Hemoglobin (Scapharca)
    
    # ===========================================
    # BACTERIOFERRITINS / FERRITIN-LIKE
    # ===========================================
    "1BFR",  # Bacterioferritin
    "1NF4",  # Bacterioferritin
    "1D9O",  # Rubrerythrin (ferritin-like)
    "1LKO",  # Dps protein (ferritin-like)
    
    # ===========================================
    # ACYL CARRIER PROTEINS (small alpha bundle)
    # ===========================================
    "1ACP",  # Acyl carrier protein
    "1L0I",  # Acyl carrier protein
    "1HY8",  # Acyl carrier protein
    
    # ===========================================
    # HELICAL CYTOKINES (additional)
    # ===========================================
    "1ALU",  # Interleukin-8 monomer (alpha form)
    "1F45",  # LIF (leukemia inhibitory factor)
    "1CNT",  # CNTF (ciliary neurotrophic factor)
    "1PGR",  # Placental growth hormone
    
    # ===========================================
    # Z-DNA BINDING / HELICAL DNA-BINDING
    # ===========================================
    "1QBJ",  # Z-DNA binding domain
    "1J75",  # Z-alpha domain
    "1SRA",  # SRP receptor alpha (helical)
    
    # ===========================================
    # HELICAL BUNDLE TOXINS
    # ===========================================
    "1CLV",  # Colicin (helical bundle)
    "1A87",  # Colicin domain
    "1CII",  # Colicin immunity protein
    
    # ===========================================
    # SMALL HELICAL DOMAINS
    # ===========================================
    "1BW6",  # Helical domain
    "1C1K",  # Helical domain
    "1BG8",  # Alpha domain
    "1BHH",  # Helical bundle
    "1CEI",  # Colicin E9 immunity
    "1C52",  # Small alpha protein
    "1DP3",  # Helical protein
    
    # ===========================================
    # ADDITIONAL EF-HAND RELATED
    # ===========================================
    "1PSR",  # Polcalcin (EF-hand)
    "2PVB",  # Parvalbumin
    "1RRO",  # Calcyclin (S100)
    "1K9P",  # S100A6
    
    # ===========================================
    # REPRESSOR/HTH ADDITIONAL
    # ===========================================
    "1PER",  # Purine repressor
    "1WET",  # TetR family repressor
    "1SAX",  # SAP-1 (Ets domain),
    # ===========================================
    # ADDITIONAL GLOBINS
    # ===========================================
    "1CPW",  # Cytoglobin
    "1URY",  # Hemoglobin
    "1HBR",  # Hemoglobin R state
    "1OUT",  # Hemoglobin
    "1J41",  # Hemoglobin variant
    
    # ===========================================
    # MORE FERRITIN-LIKE
    # ===========================================
    "1JGC",  # Ferritin-like
    "1QGH",  # Rubrerythrin
    "1S3Q",  # Dps-like
    
    # ===========================================
    # CYTOCHROME / HEME BINDING
    # ===========================================
    "1DW0",  # Cytochrome
    "1FS7",  # Cytochrome
    "1J8Q",  # Cytochrome
    "1M70",  # Cytochrome
    "2B0Z",  # Cytochrome
    "1FGJ",  # Cytochrome c'
    
    # ===========================================
    # SMALL HELICAL BUNDLES
    # ===========================================
    "1RIJ",  # Helical bundle
    "1KOD",  # Helical protein
    "1T0G",  # Helical bundle
    "1TM1",  # Helical bundle
    "1K8U",  # Designed helical
    "1S2O",  # Alpha helical
    "1OAI",  # Alpha bundle
    "1R6J",  # Helical protein
    "1OPD",  # Alpha helical
    
    # ===========================================
    # EF-HAND / CALCIUM BINDING
    # ===========================================
    "1CDN",  # Calbindin D9k
    "4ICB",  # Calbindin
    "1IG5",  # S100 protein
    "1IRJ",  # Centrin
    "1J7O",  # EF-hand protein
    "1NWA",  # EF-hand
    
    # ===========================================
    # COILED-COIL / HELICAL BUNDLES
    # ===========================================
    "1KDD",  # Coiled coil
    "1T6F",  # Coiled coil
    "2CCE",  # Coiled coil
    "1P9I",  # Coiled coil
    
    # ===========================================
    # MISC SMALL ALPHA
    # ===========================================
    "1C75",  # Small alpha
    "1A32",  # Helical protein
    "1G6U",  # Alpha helical
    "1JWE",  # Helical bundle
    "1N88",  # Alpha protein
    "1OJ7",  # Small helical
]


def download_pdb(pdb_id, output_dir="pdbs"):
    """Download a PDB file from RCSB."""
    os.makedirs(output_dir, exist_ok=True)
    
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"  Failed to download {pdb_id}")
        return None
    
    filepath = os.path.join(output_dir, f"{pdb_id}.pdb")
    with open(filepath, 'w') as f:
        f.write(response.text)
    
    return filepath


def get_sequence_from_pdb_file(pdb_path, chain="A"):
    """Extract sequence from PDB file."""
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import protein_letters_3to1
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    
    for model in structure:
        for ch in model:
            if ch.id == chain:
                seq = ""
                for residue in ch:
                    if residue.id[0] == " ":  # Standard residue
                        resname = residue.resname
                        if resname in protein_letters_3to1:
                            seq += protein_letters_3to1[resname]
                return seq
    
    # If chain A not found, try first chain
    for model in structure:
        for ch in model:
            seq = ""
            for residue in ch:
                if residue.id[0] == " ":
                    resname = residue.resname
                    if resname in protein_letters_3to1:
                        seq += protein_letters_3to1[resname]
            if seq:
                return seq
    
    return None


def verify_no_beta_strands(pdb_path):
    """
    Use DSSP to verify the structure has no beta strands.
    Returns (is_all_alpha, ss_counts)
    """
    try:
        from Bio.PDB import PDBParser
        from Bio.PDB.DSSP import DSSP
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)
        model = structure[0]
        
        dssp = DSSP(model, pdb_path, dssp='mkdssp')
        
        ss_elements = [res[2] for res in dssp]
        
        # Check for beta strands (E) or beta bridges (B)
        has_beta = any(ss in ['E', 'B'] for ss in ss_elements)
        
        # Count secondary structure
        n_helix = sum(1 for ss in ss_elements if ss in ['H', 'G', 'I'])
        n_coil = sum(1 for ss in ss_elements if ss in ['T', 'S', '-', 'C', ' '])
        n_beta = sum(1 for ss in ss_elements if ss in ['E', 'B'])
        
        return not has_beta, {
            "helix": n_helix, 
            "coil": n_coil, 
            "beta": n_beta,
            "total": len(ss_elements)
        }
        
    except Exception as e:
        print(f"  DSSP failed: {e}")
        return None, None


def main():
    output_dir = "all_alpha_pdbs"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading {len(KNOWN_ALL_ALPHA)} candidate all-alpha proteins...")
    
    verified_proteins = []
    
    for i, pdb_id in enumerate(KNOWN_ALL_ALPHA):
        print(f"[{i+1}/{len(KNOWN_ALL_ALPHA)}] Processing {pdb_id}...", end=" ")
        
        # Download
        pdb_path = download_pdb(pdb_id, output_dir)
        if not pdb_path:
            continue
        
        # Verify no beta strands
        is_all_alpha, ss_counts = verify_no_beta_strands(pdb_path)
        
        if is_all_alpha is None:
            print(f"Could not verify, skipping")
            os.remove(pdb_path)
            continue
        
        if not is_all_alpha:
            print(f"Has {ss_counts['beta']} beta residues, removing")
            os.remove(pdb_path)
            continue
        
        # Get sequence
        seq = get_sequence_from_pdb_file(pdb_path)
        if seq and 30 <= len(seq) <= 400:
            helix_frac = ss_counts['helix'] / ss_counts['total'] if ss_counts['total'] > 0 else 0
            print(f"✓ len={len(seq)}, {ss_counts['helix']} helix ({helix_frac:.0%})")
            
            verified_proteins.append({
                "pdb_id": pdb_id,
                "sequence": seq,
                "length": len(seq),
                "helix_residues": ss_counts['helix'],
                "coil_residues": ss_counts['coil'],
                "helix_fraction": helix_frac,
            })
        else:
            if seq and len(seq) > 400:
                print(f"Sequence too long ({len(seq)} > 400), removing")
            elif seq and len(seq) < 30:
                print(f"Sequence too short ({len(seq)} < 30), removing")
            else:
                print(f"Sequence not found, removing")
            os.remove(pdb_path)
        
        time.sleep(0.1)
    
    # Save summary
    if verified_proteins:
        df = pd.DataFrame(verified_proteins)
        csv_path = os.path.join(output_dir, "all_alpha_proteins.csv")
        df.to_csv(csv_path, index=False)
        
        print(f"\n{'='*60}")
        print(f"Downloaded {len(verified_proteins)} verified all-alpha proteins")
        print(f"Saved to {output_dir}/")
        print(f"Summary CSV: {csv_path}")
        print(f"\nProteins:")
        print(df[["pdb_id", "length", "helix_residues", "helix_fraction"]].to_string())
        
        return df
    else:
        print("\nNo proteins passed verification!")
        return pd.DataFrame()


if __name__ == "__main__":
    df = main()