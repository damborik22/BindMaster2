"""Refolding orchestrator: runs designs through all engines.

Handles failures gracefully — if one engine fails, results from others
are still used. Computes ipSAE from PAE matrices using metrics/ipsae.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from bm2_evaluator.core.models import (
    EngineResult,
    EvalConfig,
    EvaluationResult,
    IngestedDesign,
)
from bm2_evaluator.metrics.ipsae import compute_ipsae
from bm2_evaluator.metrics.pae import load_pae_matrix, save_pae_matrix
from bm2_evaluator.metrics.plddt import normalize_plddt as _normalize_plddt
from bm2_evaluator.refolding.base import RefoldingEngine, WorkerOutput
from bm2_evaluator.refolding.monomer import MonomerValidator, MonomerValidationResult

logger = logging.getLogger(__name__)


class RefoldingOrchestrator:
    """Runs a design through all configured refolding engines.

    Pipeline per design:
        1. Refold complex through each engine (Boltz2, AF2)
        2. Load PAE matrix, compute ipSAE
        3. Extract pLDDT/ipTM, normalize
        4. Run monomer validation (optional)
        5. Aggregate into EvaluationResult
    """

    def __init__(
        self,
        engines: list[RefoldingEngine],
        config: EvalConfig,
        monomer_engine: Optional[RefoldingEngine] = None,
    ):
        """Initialize orchestrator.

        Args:
            engines: List of refolding engines to use.
            config: Evaluation configuration.
            monomer_engine: Engine for monomer validation.
                            Defaults to first engine in list.
        """
        self.engines = {e.name: e for e in engines}
        self.config = config
        self.monomer_validator = MonomerValidator(
            engine=monomer_engine or engines[0],
            rmsd_threshold=config.monomer_rmsd_threshold,
        )

    def evaluate_design(
        self,
        design: IngestedDesign,
        output_dir: Path,
        run_monomer: bool = True,
    ) -> EvaluationResult:
        """Full evaluation of one design through all engines.

        Args:
            design: The ingested design to evaluate.
            output_dir: Base output directory for this design.
            run_monomer: Whether to run monomer validation.

        Returns:
            EvaluationResult with all engine results and monomer validation.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Evaluating {design.design_id} from {design.source_tool.value}"
        )

        eval_result = EvaluationResult(design=design)

        # Run each engine
        for engine_name, engine in self.engines.items():
            engine_dir = output_dir / engine_name

            try:
                worker_output = engine.refold_complex(
                    binder_seq=design.binder_sequence,
                    target_seq=design.target_sequence,
                    output_dir=engine_dir,
                )

                if not worker_output.success:
                    logger.error(
                        f"  {engine_name} failed: {worker_output.error}"
                    )
                    continue

                # Build EngineResult from worker output
                engine_result = self._build_engine_result(
                    worker_output, engine_dir
                )
                eval_result.engine_results[engine_name] = engine_result

                logger.info(
                    f"  {engine_name}: iptm={engine_result.iptm:.3f}, "
                    f"ipsae_min={engine_result.ipsae_min:.3f}"
                )

            except Exception as e:
                logger.error(f"  {engine_name} error: {e}")
                continue

        if not eval_result.engine_results:
            logger.error(f"All engines failed for {design.design_id}")
            return eval_result

        # Monomer validation
        if run_monomer:
            self._run_monomer_validation(design, eval_result, output_dir)

        return eval_result

    def _build_engine_result(
        self, worker_output: WorkerOutput, engine_dir: Path
    ) -> EngineResult:
        """Convert WorkerOutput to EngineResult with ipSAE computation."""
        # Load PAE matrix
        pae_path = engine_dir / worker_output.pae_matrix_file
        pae_matrix = load_pae_matrix(pae_path)

        # Get chain slices based on worker-reported chain order
        binder_slice = worker_output.get_binder_slice()
        target_slice = worker_output.get_target_slice()

        # Compute ipSAE using our validated implementation
        ipsae_result = compute_ipsae(
            pae_matrix=pae_matrix,
            binder_slice=binder_slice,
            target_slice=target_slice,
            cutoff=self.config.pae_cutoff,  # 15A default (Dunbrack)
        )

        # Normalize pLDDT
        scale = "0-1" if worker_output.plddt_scale_max <= 1.0 else "0-100"
        _, plddt_binder_norm = _normalize_plddt(
            worker_output.plddt_binder_mean, scale
        )

        # Compute interaction PAE mean
        pae_bt = pae_matrix[binder_slice, target_slice]
        pae_tb = pae_matrix[target_slice, binder_slice]
        pae_interaction = float((np.mean(pae_bt) + np.mean(pae_tb)) / 2.0)

        # Compute binder-internal PAE mean
        pae_binder = pae_matrix[binder_slice, binder_slice]
        pae_binder_mean = float(np.mean(pae_binder))

        # Standardize PAE storage as .npy in our convention
        standardized_pae_path = engine_dir / "pae.npy"
        if not standardized_pae_path.exists():
            save_pae_matrix(pae_matrix, standardized_pae_path)

        structure_path = engine_dir / worker_output.structure_file

        return EngineResult(
            engine=worker_output.engine,
            pae_matrix_path=standardized_pae_path,
            pae_matrix_shape=pae_matrix.shape,
            bt_ipsae=ipsae_result.bt_ipsae,
            tb_ipsae=ipsae_result.tb_ipsae,
            ipsae_min=ipsae_result.ipsae_min,
            ipsae_max=ipsae_result.ipsae_max,
            iptm=worker_output.iptm,
            ptm=worker_output.ptm,
            plddt_binder_mean_raw=worker_output.plddt_binder_mean,
            plddt_binder_mean_norm=plddt_binder_norm,
            plddt_target_mean_raw=worker_output.plddt_target_mean,
            pae_interaction_mean=pae_interaction,
            pae_binder_mean=pae_binder_mean,
            refolded_structure_path=structure_path,
            n_interface_contacts=0,  # Computed in Step 3 if needed
        )

    def _run_monomer_validation(
        self,
        design: IngestedDesign,
        eval_result: EvaluationResult,
        output_dir: Path,
    ) -> None:
        """Run monomer validation using the first successful engine's structure."""
        # Use the first engine's refolded structure as reference
        first_engine_result = next(iter(eval_result.engine_results.values()))
        complex_structure = first_engine_result.refolded_structure_path

        # Determine binder chain ID in the refolded structure
        # Workers produce structures where the binder position depends
        # on the engine's chain ordering convention
        binder_chain = design.binder_chain

        try:
            monomer_result = self.monomer_validator.validate(
                binder_seq=design.binder_sequence,
                complex_structure_path=complex_structure,
                binder_chain=binder_chain,
                output_dir=output_dir / "monomer",
            )

            eval_result.monomer_rmsd = monomer_result.monomer_rmsd
            eval_result.monomer_passes = monomer_result.passes_validation

            logger.info(
                f"  Monomer: RMSD={monomer_result.monomer_rmsd:.2f}A "
                f"({'PASS' if monomer_result.passes_validation else 'FAIL'})"
            )

        except Exception as e:
            logger.warning(f"  Monomer validation failed: {e}")
