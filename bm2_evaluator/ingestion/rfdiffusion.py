"""RFdiffusion + LigandMPNN output ingestor.

Expected structure:
    output_dir/
    +-- *.pdb               Backbone PDB files from RFdiffusion
    +-- *.trb               Pickle files with diffusion metadata
    +-- seqs/
        +-- *.fa            FASTA files from LigandMPNN

Chain convention: chain A = binder (designed), chain B = target.
NOTE: This is OPPOSITE to BindCraft convention.

The .trb files are Python pickle dictionaries with RFdiffusion metadata.
FASTA headers contain LigandMPNN scores:
    >design_0, score=1.234, global_score=1.234, ...

PDB files are backbone-only (N, CA, C, O). Sidechains come from
LigandMPNN sequences.
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Optional

from bm2_evaluator.core.models import IngestedDesign, SourceTool
from bm2_evaluator.ingestion.base import DesignIngestor

logger = logging.getLogger(__name__)


class RFdiffusionIngestor(DesignIngestor):
    """Parse RFdiffusion + LigandMPNN output directory."""

    @property
    def tool_name(self) -> str:
        return "rfdiffusion"

    def ingest(
        self,
        output_dir: Path,
        target_chain: str = "B",  # RFdiffusion: target is chain B
        binder_chain: str = "A",  # RFdiffusion: binder is chain A
        **kwargs,
    ) -> list[IngestedDesign]:
        output_dir = Path(output_dir)

        # Find PDB backbone files
        pdb_files = sorted(output_dir.glob("*.pdb"))
        if not pdb_files:
            raise FileNotFoundError(f"No PDB files found in {output_dir}")

        # Load FASTA sequences from LigandMPNN
        fasta_data = self._load_fasta_sequences(output_dir)

        # Load .trb metadata
        trb_data = self._load_trb_files(output_dir)

        designs = []
        for pdb_path in pdb_files:
            backbone_id = pdb_path.stem

            try:
                seqs = self._extract_sequences_from_pdb(
                    pdb_path, [target_chain, binder_chain]
                )
            except Exception as e:
                logger.warning(f"Failed to parse {pdb_path}: {e}")
                continue

            target_seq = seqs.get(target_chain, "")

            # Prefer LigandMPNN-designed sequences over backbone PDB sequences
            # (backbone PDBs have GLY/ALA stubs, not real sequences)
            fasta_seqs = fasta_data.get(backbone_id, [])

            if fasta_seqs:
                # Each backbone may have multiple MPNN sequences
                for i, (mpnn_seq, mpnn_metrics) in enumerate(fasta_seqs):
                    design_id = (
                        f"{backbone_id}_mpnn_{i}" if len(fasta_seqs) > 1
                        else backbone_id
                    )

                    tool_metrics = dict(mpnn_metrics)

                    # Add trb metadata if available
                    if backbone_id in trb_data:
                        for k, v in trb_data[backbone_id].items():
                            tool_metrics[f"trb_{k}"] = v

                    design = IngestedDesign(
                        design_id=design_id,
                        source_tool=SourceTool.RFDIFFUSION,
                        binder_sequence=mpnn_seq,
                        binder_chain=binder_chain,
                        target_sequence=target_seq,
                        target_chain=target_chain,
                        binder_length=len(mpnn_seq),
                        target_length=len(target_seq),
                        complex_structure_path=pdb_path,
                        tool_metrics=tool_metrics,
                    )

                    warnings = self._validate_design(design)
                    for w in warnings:
                        logger.warning(w)

                    designs.append(design)
            else:
                # No FASTA available; use backbone PDB sequence
                binder_seq = seqs.get(binder_chain, "")
                if not target_seq or not binder_seq:
                    logger.warning(f"Missing chains in {pdb_path}")
                    continue

                tool_metrics = {}
                if backbone_id in trb_data:
                    for k, v in trb_data[backbone_id].items():
                        tool_metrics[f"trb_{k}"] = v

                design = IngestedDesign(
                    design_id=backbone_id,
                    source_tool=SourceTool.RFDIFFUSION,
                    binder_sequence=binder_seq,
                    binder_chain=binder_chain,
                    target_sequence=target_seq,
                    target_chain=target_chain,
                    binder_length=len(binder_seq),
                    target_length=len(target_seq),
                    complex_structure_path=pdb_path,
                    tool_metrics=tool_metrics,
                    metadata={"note": "backbone_only_no_mpnn_sequence"},
                )
                designs.append(design)

        logger.info(
            f"Ingested {len(designs)} RFdiffusion designs from {output_dir}"
        )
        return designs

    def _load_fasta_sequences(
        self, output_dir: Path
    ) -> dict[str, list[tuple[str, dict[str, float]]]]:
        """Load LigandMPNN FASTA files.

        Returns: {backbone_id: [(sequence, {score: val, ...}), ...]}
        """
        result: dict[str, list[tuple[str, dict[str, float]]]] = {}

        seqs_dir = output_dir / "seqs"
        if not seqs_dir.is_dir():
            # Try flat directory
            fa_files = list(output_dir.glob("*.fa")) + list(
                output_dir.glob("*.fasta")
            )
        else:
            fa_files = list(seqs_dir.glob("*.fa")) + list(
                seqs_dir.glob("*.fasta")
            )

        for fa_path in fa_files:
            backbone_id = fa_path.stem
            entries = self._parse_fasta(fa_path)
            if entries:
                result[backbone_id] = entries

        return result

    def _parse_fasta(
        self, fa_path: Path
    ) -> list[tuple[str, dict[str, float]]]:
        """Parse a LigandMPNN FASTA file.

        Returns list of (sequence, metrics_dict) tuples.
        Skips the first entry (native/input sequence).
        """
        entries = []
        current_header = ""
        current_seq_lines: list[str] = []

        with open(fa_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_header and current_seq_lines:
                        entries.append(
                            (
                                "".join(current_seq_lines),
                                self._parse_fasta_header(current_header),
                            )
                        )
                    current_header = line[1:]
                    current_seq_lines = []
                elif line:
                    current_seq_lines.append(line)

            if current_header and current_seq_lines:
                entries.append(
                    (
                        "".join(current_seq_lines),
                        self._parse_fasta_header(current_header),
                    )
                )

        # Skip first entry (native/input sequence)
        if len(entries) > 1:
            return entries[1:]
        return entries

    def _parse_fasta_header(self, header: str) -> dict[str, float]:
        """Parse key=value pairs from a LigandMPNN FASTA header."""
        metrics: dict[str, float] = {}
        # Match patterns like score=1.234 or global_score=1.234
        for match in re.finditer(r"(\w+)=([\d.eE+-]+)", header):
            key, val = match.groups()
            try:
                metrics[key] = float(val)
            except ValueError:
                pass
        return metrics

    def _load_trb_files(
        self, output_dir: Path
    ) -> dict[str, dict[str, float]]:
        """Load .trb pickle metadata files.

        Extracts numeric scalar values only.
        """
        result: dict[str, dict[str, float]] = {}

        for trb_path in output_dir.glob("*.trb"):
            backbone_id = trb_path.stem
            try:
                with open(trb_path, "rb") as f:
                    data = pickle.load(f)
                metrics: dict[str, float] = {}
                for k, v in data.items():
                    if isinstance(v, (int, float)):
                        metrics[k] = float(v)
                result[backbone_id] = metrics
            except Exception as e:
                logger.warning(f"Failed to load {trb_path}: {e}")

        return result
