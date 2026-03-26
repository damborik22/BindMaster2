#!/usr/bin/env python3
"""Target analyzer for BM2 loss tuning.

Analyzes a target PDB file and produces target_analysis.json with:
- Basic structure info (chains, residues, secondary structure)
- SASA per residue (for binding site identification)
- Potential binding sites (high-SASA surface patches)
- Glycosylation sequons (N-X-S/T)
- Oligomeric state
- Difficulty assessment
- Recommended starting strategy

Usage:
    python agent/target_analyzer.py --pdb target.pdb --chain A --output target_analysis.json
"""

import argparse
import json
import sys
from pathlib import Path


def analyze_target(pdb_path: str, chain_id: str = "A") -> dict:
    """Analyze a target PDB file.

    Returns a dict with structural analysis results.
    """
    from Bio.PDB import PDBParser
    from Bio.PDB.DSSP import DSSP

    pdb_path = Path(pdb_path)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("target", str(pdb_path))
    model = structure[0]

    # Find the target chain
    chain = None
    for c in model:
        if c.id == chain_id:
            chain = c
            break
    if chain is None:
        chains_available = [c.id for c in model]
        raise ValueError(
            f"Chain {chain_id} not found. Available: {chains_available}"
        )

    # Basic info
    residues = [r for r in chain if r.id[0] == " "]  # exclude hetero
    n_residues = len(residues)
    sequence = ""
    from Bio.Data.IUPACData import protein_letters_3to1
    _3to1 = {k.upper(): v.upper() for k, v in protein_letters_3to1.items()}
    for r in residues:
        try:
            sequence += _3to1[r.resname.strip().upper()]
        except KeyError:
            sequence += "X"

    # All chains in the model
    all_chains = [c.id for c in model]
    n_chains = len(all_chains)
    identical_chains = sum(
        1 for c in model
        if c.id != chain_id and len(list(c.get_residues())) == n_residues
    )

    # Oligomeric state
    if n_chains == 1:
        oligomeric = "monomer"
    elif identical_chains >= 1:
        oligomeric = f"homo-{n_chains}mer (likely)"
    else:
        oligomeric = f"hetero-{n_chains}mer"

    # SASA
    sasa_per_residue = {}
    try:
        import freesasa
        fs_structure = freesasa.Structure(str(pdb_path))
        result = freesasa.calc(fs_structure)
        # Per-residue SASA from freesasa
        for i, r in enumerate(residues):
            res_key = f"{chain_id}{r.id[1]}"
            # Simple approximation: sum atom SASA for this residue
            sasa_per_residue[res_key] = 0.0
        # Use selection-based approach
        for i, r in enumerate(residues):
            resnum = r.id[1]
            sel = freesasa.selectArea(
                {f"res{resnum}": f"chain {chain_id} and resi {resnum}"},
                fs_structure, result
            )
            sasa_per_residue[f"{chain_id}{resnum}"] = sel.get(f"res{resnum}", 0.0)
    except (ImportError, Exception):
        # Fallback: estimate from B-factors (higher B = more exposed)
        for r in residues:
            atoms = list(r.get_atoms())
            avg_b = sum(a.bfactor for a in atoms) / max(len(atoms), 1)
            sasa_per_residue[f"{chain_id}{r.id[1]}"] = avg_b

    # Identify high-SASA residues (potential binding sites)
    sasa_values = list(sasa_per_residue.values())
    if sasa_values:
        mean_sasa = sum(sasa_values) / len(sasa_values)
        high_sasa_residues = [
            k for k, v in sasa_per_residue.items()
            if v > mean_sasa * 1.5
        ]
    else:
        high_sasa_residues = []

    # Glycosylation sequons (N-X-S/T where X != P)
    glycosylation_sites = []
    for i in range(len(sequence) - 2):
        if sequence[i] == "N" and sequence[i + 1] != "P" and sequence[i + 2] in ("S", "T"):
            resnum = residues[i].id[1]
            glycosylation_sites.append(f"{chain_id}{resnum}")

    # Secondary structure composition (simple estimation)
    # Count helix/sheet/coil from backbone geometry if DSSP not available
    ss_composition = {"helix": 0, "sheet": 0, "coil": 0}
    try:
        dssp = DSSP(model, str(pdb_path), dssp="mkdssp")
        for key in dssp:
            ss = dssp[key][2]
            if ss in ("H", "G", "I"):
                ss_composition["helix"] += 1
            elif ss in ("E", "B"):
                ss_composition["sheet"] += 1
            else:
                ss_composition["coil"] += 1
    except Exception:
        # Can't run DSSP, estimate from sequence length
        ss_composition["coil"] = n_residues

    total_ss = sum(ss_composition.values()) or 1
    ss_fractions = {k: v / total_ss for k, v in ss_composition.items()}

    # Surface characterization
    # Count charged (D,E,K,R,H), hydrophobic (A,V,I,L,M,F,W,P), polar (S,T,N,Q,Y,C,G)
    charged = sum(1 for aa in sequence if aa in "DEKRH")
    hydrophobic = sum(1 for aa in sequence if aa in "AVILMFWP")
    polar = sum(1 for aa in sequence if aa in "STNQYCG")
    surface_type = "balanced"
    if charged / max(n_residues, 1) > 0.35:
        surface_type = "highly charged"
    elif hydrophobic / max(n_residues, 1) > 0.45:
        surface_type = "hydrophobic"

    # Difficulty assessment
    difficulty = 0.3  # base
    if n_residues < 50:
        difficulty += 0.2  # small targets are harder
    if n_residues > 300:
        difficulty += 0.1
    if glycosylation_sites:
        difficulty += 0.1 * min(len(glycosylation_sites), 3)
    if oligomeric != "monomer":
        difficulty += 0.1
    difficulty = min(difficulty, 1.0)

    # Strategy recommendation
    if difficulty < 0.3:
        strategy = "Standard Escalante defaults. 80 AA binder, no epitope specification."
    elif difficulty < 0.6:
        strategy = "Increase binder length to 120. Consider hotspot residues if known."
    else:
        strategy = (
            "Difficult target. Try 150-200 AA binders. Use target PDB template for refolding. "
            "Consider beam_search in Complexa."
        )

    return {
        "target_name": pdb_path.stem,
        "pdb_path": str(pdb_path),
        "chain": chain_id,
        "sequence": sequence,
        "n_residues": n_residues,
        "all_chains": all_chains,
        "oligomeric_state": oligomeric,
        "ss_composition": ss_composition,
        "ss_fractions": {k: round(v, 2) for k, v in ss_fractions.items()},
        "surface_type": surface_type,
        "charged_fraction": round(charged / max(n_residues, 1), 2),
        "hydrophobic_fraction": round(hydrophobic / max(n_residues, 1), 2),
        "high_sasa_residues": high_sasa_residues[:20],  # top 20
        "glycosylation_sites": glycosylation_sites,
        "difficulty": round(difficulty, 2),
        "strategy": strategy,
        "n_high_sasa": len(high_sasa_residues),
    }


def fill_strategy_template(
    strategy_path: Path,
    analysis: dict,
    output_path: Path | None = None,
) -> str:
    """Fill strategy.md placeholders with target analysis results.

    Replaces {target_name}, {pdb_code}, {chain}, {n_residues}, etc.
    """
    template = Path(strategy_path).read_text()

    replacements = {
        "{target_name}": analysis.get("target_name", "unknown"),
        "{pdb_code}": analysis.get("target_name", "unknown"),
        "{organism}": "unknown",  # would need UniProt lookup
        "{chain}": analysis.get("chain", "A"),
        "{n_residues}": str(analysis.get("n_residues", "?")),
        "{oligomeric_state}": analysis.get("oligomeric_state", "unknown"),
        "{surface_summary}": (
            f"{analysis.get('surface_type', 'unknown')} surface, "
            f"{analysis.get('charged_fraction', 0):.0%} charged, "
            f"{analysis.get('hydrophobic_fraction', 0):.0%} hydrophobic, "
            f"SS: {analysis.get('ss_fractions', {}).get('helix', 0):.0%} helix / "
            f"{analysis.get('ss_fractions', {}).get('sheet', 0):.0%} sheet"
        ),
        "{difficulty}": f"{analysis.get('difficulty', 0.3):.2f}",
        "{strategy}": analysis.get("strategy", "Standard defaults"),
    }

    filled = template
    for key, value in replacements.items():
        filled = filled.replace(key, value)

    if output_path:
        Path(output_path).write_text(filled)

    return filled


def main():
    parser = argparse.ArgumentParser(description="BM2 Target Analyzer")
    parser.add_argument("--pdb", required=True, help="Target PDB file")
    parser.add_argument("--chain", default="A", help="Target chain (default: A)")
    parser.add_argument("--output", default="target_analysis.json", help="Output JSON path")
    parser.add_argument(
        "--strategy-template", default=None,
        help="Path to strategy.md template to fill with analysis results"
    )
    parser.add_argument(
        "--strategy-output", default=None,
        help="Path to write filled strategy.md (default: strategy_filled.md)"
    )
    args = parser.parse_args()

    analysis = analyze_target(args.pdb, args.chain)

    output = json.dumps(analysis, indent=2)
    Path(args.output).write_text(output)
    print(f"Analysis saved to {args.output}")
    print(f"  Target: {analysis['target_name']} ({analysis['n_residues']} residues)")
    print(f"  Oligomeric: {analysis['oligomeric_state']}")
    print(f"  Surface: {analysis['surface_type']}")
    print(f"  Difficulty: {analysis['difficulty']}")
    print(f"  Strategy: {analysis['strategy']}")

    if args.strategy_template:
        strategy_out = args.strategy_output or "strategy_filled.md"
        fill_strategy_template(
            Path(args.strategy_template), analysis, Path(strategy_out)
        )
        print(f"  Filled strategy: {strategy_out}")


if __name__ == "__main__":
    main()
