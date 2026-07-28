"""
Main controller for OpenEvolve
"""

import asyncio
import logging
import os
import shutil
import signal
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from openevolve.config import Config, load_config
from openevolve.database import Program, ProgramDatabase
from openevolve.evaluator import Evaluator
from openevolve.evolution_trace import EvolutionTracer
from openevolve.llm.ensemble import LLMEnsemble
from openevolve.process_parallel import ProcessParallelController
from openevolve.prompt.sampler import PromptSampler
from openevolve.utils.code_utils import extract_code_language
from openevolve.utils.format_utils import format_improvement_safe, format_metrics_safe

logger = logging.getLogger(__name__)


class OpenEvolve:
    """
    Main controller for OpenEvolve

    Orchestrates the evolution process, coordinating between the prompt sampler,
    LLM ensemble, evaluator, and program database.

    Features:
    - Tracks the absolute best program across evolution steps
    - Ensures the best solution is not lost during the MAP-Elites process
    - Always includes the best program in the selection process for inspiration
    - Maintains detailed logs and metadata about improvements
    """

    def __init__(
        self,
        initial_program_path: str,
        evaluation_file: str,
        config: Config,
        output_dir: Optional[str] = None,
    ):
        # Load configuration (loaded in main_async)
        self.config = config

        # Set up output directory
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(initial_program_path), "openevolve_output"
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # Set up logging
        self._setup_logging()

        # Manual mode queue lives in <openevolve_output>/manual_tasks_queue
        self._setup_manual_mode_queue()

        # Set random seed for reproducibility if specified
        if self.config.random_seed is not None:
            import hashlib
            import random

            import numpy as np

            # Set global random seeds
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)

            # Create hash-based seeds for different components. md5 is used only to
            # derive a deterministic RNG seed from the configured seed, not for any
            # security purpose; usedforsecurity=False documents that intent.
            base_seed = str(self.config.random_seed).encode("utf-8")
            llm_seed = int(
                hashlib.md5(base_seed + b"llm", usedforsecurity=False).hexdigest()[:8], 16
            ) % (2**31)

            # Propagate seed to LLM configurations
            self.config.llm.random_seed = llm_seed
            for model_cfg in self.config.llm.models:
                if not hasattr(model_cfg, "random_seed") or model_cfg.random_seed is None:
                    model_cfg.random_seed = llm_seed
            for model_cfg in self.config.llm.evaluator_models:
                if not hasattr(model_cfg, "random_seed") or model_cfg.random_seed is None:
                    model_cfg.random_seed = llm_seed

            logger.info(f"Set random seed to {self.config.random_seed} for reproducibility")
            logger.debug(f"Generated LLM seed: {llm_seed}")

        # Load initial program
        self.initial_program_path = initial_program_path
        self.initial_program_code = self._load_initial_program()
        if not self.config.language:
            self.config.language = extract_code_language(self.initial_program_code)

        # Extract file extension from initial program
        self.file_extension = os.path.splitext(initial_program_path)[1]
        if not self.file_extension:
            # Default to .py if no extension found
            self.file_extension = ".py"
        else:
            # Make sure it starts with a dot
            if not self.file_extension.startswith("."):
                self.file_extension = f".{self.file_extension}"

        # Set the file_suffix in config (can be overridden in YAML)
        if not hasattr(self.config, "file_suffix") or self.config.file_suffix == ".py":
            self.config.file_suffix = self.file_extension

        # Initialize components
        self.llm_ensemble = LLMEnsemble(self.config.llm.models)
        self.llm_evaluator_ensemble = LLMEnsemble(self.config.llm.evaluator_models)

        self.prompt_sampler = PromptSampler(self.config.prompt)
        self.evaluator_prompt_sampler = PromptSampler(self.config.prompt)
        self.evaluator_prompt_sampler.set_templates("evaluator_system_message")

        # Pass random seed to database if specified
        if self.config.random_seed is not None:
            self.config.database.random_seed = self.config.random_seed

        self.config.database.novelty_llm = self.llm_ensemble
        self.database = ProgramDatabase(self.config.database)

        self.evaluator = Evaluator(
            self.config.evaluator,
            evaluation_file,
            self.llm_evaluator_ensemble,
            self.evaluator_prompt_sampler,
            database=self.database,
            suffix=Path(self.initial_program_path).suffix,
        )
        self.evaluation_file = evaluation_file

        logger.info(f"Initialized OpenEvolve with {initial_program_path}")

        # Initialize evolution tracer
        if self.config.evolution_trace.enabled:
            trace_output_path = self.config.evolution_trace.output_path
            if not trace_output_path:
                # Default to output_dir/evolution_trace.{format}
                trace_output_path = os.path.join(
                    self.output_dir, f"evolution_trace.{self.config.evolution_trace.format}"
                )

            self.evolution_tracer = EvolutionTracer(
                output_path=trace_output_path,
                format=self.config.evolution_trace.format,
                include_code=self.config.evolution_trace.include_code,
                include_prompts=self.config.evolution_trace.include_prompts,
                enabled=True,
                buffer_size=self.config.evolution_trace.buffer_size,
                compress=self.config.evolution_trace.compress,
            )
            logger.info(f"Evolution tracing enabled: {trace_output_path}")
        else:
            self.evolution_tracer = None

        # Initialize improved parallel processing components
        self.parallel_controller = None

    def _setup_logging(self) -> None:
        """Set up logging"""
        log_dir = self.config.log_dir or os.path.join(self.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        # Set up root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.log_level))

        # Add file handler
        log_file = os.path.join(log_dir, f"openevolve_{time.strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        root_logger.addHandler(file_handler)

        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        root_logger.addHandler(console_handler)

        logger.info(f"Logging to {log_file}")

    def _setup_manual_mode_queue(self) -> None:
        """
        Set up manual task queue directory if llm.manual_mode is enabled

        Queue directory is always:
          <openevolve_output>/manual_tasks_queue

        The directory is cleared on controller start so the UI shows only tasks
        from the current run (no stale tasks after restart)
        """
        if not bool(getattr(self.config.llm, "manual_mode", False)):
            return

        qdir = Path(self.output_dir).expanduser().resolve() / "manual_tasks_queue"

        # Clear stale tasks from previous runs
        if qdir.exists():
            shutil.rmtree(qdir)
        qdir.mkdir(parents=True, exist_ok=True)

        # Inject runtime-only queue dir into configs
        self.config.llm._manual_queue_dir = str(qdir)
        for model_cfg in self.config.llm.models:
            model_cfg._manual_queue_dir = str(qdir)
        for model_cfg in self.config.llm.evaluator_models:
            model_cfg._manual_queue_dir = str(qdir)

        logger.info(f"Manual mode enabled. Queue dir: {qdir}")

    def _load_initial_program(self) -> str:
        """Load the initial program from file"""
        with open(self.initial_program_path, "r") as f:
            return f.read()

    async def run(
        self,
        iterations: Optional[int] = None,
        target_score: Optional[float] = None,
        checkpoint_path: Optional[str] = None,
    ) -> Optional[Program]:
        """
        Run the evolution process with improved parallel processing

        Args:
            iterations: Maximum number of iterations (uses config if None)
            target_score: Target score to reach (continues until reached if specified)
            checkpoint_path: Path to resume from checkpoint

        Returns:
            Best program found
        """
        max_iterations = iterations or self.config.max_iterations
        # Determine starting iteration
        start_iteration = 0
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
            start_iteration = self.database.last_iteration + 1
            logger.info(f"Resuming from checkpoint at iteration {start_iteration}")
        else:
            start_iteration = self.database.last_iteration

        # Only add initial program if starting fresh (not resuming from checkpoint)
        should_add_initial = (
            start_iteration == 0
            and len(self.database.programs) == 0
            and not any(
                p.code == self.initial_program_code for p in self.database.programs.values()
            )
        )

        # Reset AetherState research notebooks at the start of each fresh run so
        # stale agendas/findings from previous experiments do not leak into this
        # run.  The files live next to the task file for easy inspection.
        if should_add_initial and "tasks/aetherstate" in self.initial_program_path:
            for stale_name in ("AGENDA.md", "FINDINGS.md"):
                stale_path = Path(self.initial_program_path).resolve().parent / stale_name
                if stale_path.exists():
                    try:
                        stale_path.unlink()
                        logger.info(f"Removed stale {stale_path.name} from previous run")
                    except OSError:
                        pass

        if should_add_initial:
            logger.info("Adding initial program to database")
            initial_program_id = str(uuid.uuid4())

            # Evaluate the initial program
            initial_metrics = await self.evaluator.evaluate_program(
                self.initial_program_code, initial_program_id
            )

            # Check for artifacts (including the research notebook) to carry into
            # the new Program's metadata.
            initial_artifacts = self.evaluator.get_pending_artifacts(initial_program_id)
            initial_metadata: Dict[str, Any] = {}
            if initial_artifacts and initial_artifacts.get("research_notebook"):
                initial_metadata["research_note"] = initial_artifacts["research_notebook"]

            initial_program = Program(
                id=initial_program_id,
                code=self.initial_program_code,
                changes_description=self.config.prompt.initial_changes_description,
                language=self.config.language,
                metrics=initial_metrics,
                iteration_found=start_iteration,
                metadata=initial_metadata,
            )

            self.database.add(initial_program)

            # Store artifacts after the program has been added so the database
            # can associate them with the program.
            if initial_artifacts:
                self.database.store_artifacts(initial_program_id, initial_artifacts)
                logger.info(f"Stored artifacts for initial program")

            # Check if combined_score is present in the metrics
            if "combined_score" not in initial_metrics:
                # Calculate average of numeric metrics
                numeric_metrics = [
                    v
                    for v in initial_metrics.values()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                ]
                if numeric_metrics:
                    avg_score = sum(numeric_metrics) / len(numeric_metrics)
                    logger.warning(
                        f"⚠️  No 'combined_score' metric found in evaluation results. "
                        f"Using average of all numeric metrics ({avg_score:.4f}) for evolution guidance. "
                        f"For better evolution results, please modify your evaluator to return a 'combined_score' "
                        f"metric that properly weights different aspects of program performance."
                    )
        else:
            logger.info(
                f"Skipping initial program addition (resuming from iteration {start_iteration} "
                f"with {len(self.database.programs)} existing programs)"
            )

        # Initialize improved parallel processing
        try:
            self.parallel_controller = ProcessParallelController(
                self.config,
                self.evaluation_file,
                self.database,
                self.evolution_tracer,
                file_suffix=self.config.file_suffix,
            )

            # Set up signal handlers for graceful shutdown
            def signal_handler(signum, frame):
                logger.info(f"Received signal {signum}, initiating graceful shutdown...")
                self.parallel_controller.request_shutdown()

                # Set up a secondary handler for immediate exit if user presses Ctrl+C again
                def force_exit_handler(signum, frame):
                    logger.info("Force exit requested - terminating immediately")
                    import sys

                    sys.exit(0)

                signal.signal(signal.SIGINT, force_exit_handler)

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            self.parallel_controller.start()

            # When starting from iteration 0, we've already done the initial program evaluation
            # So we need to adjust the start_iteration for the actual evolution
            evolution_start = start_iteration
            evolution_iterations = max_iterations

            # If we just added the initial program at iteration 0, start evolution from iteration 1
            if should_add_initial and start_iteration == 0:
                evolution_start = 1
                # User expects max_iterations evolutionary iterations AFTER the initial program
                # So we don't need to reduce evolution_iterations

            # Run evolution with improved parallel processing and checkpoint callback
            await self._run_evolution_with_checkpoints(
                evolution_start, evolution_iterations, target_score
            )

        finally:
            # Clean up parallel processing resources
            if self.parallel_controller:
                self.parallel_controller.stop()
                self.parallel_controller = None

            # Make sure any in-flight meta-analysis task is awaited before
            # shutting down so the event loop does not warn about a dangling
            # coroutine and any pending agenda/finding writes are flushed.
            pending = getattr(self, "_pending_meta_analysis", None)
            if pending is not None and not pending.done():
                try:
                    await asyncio.wait_for(pending, timeout=30.0)
                except (asyncio.TimeoutError, Exception) as e:
                    logger.debug(f"Meta-analysis task cancelled during shutdown: {e}")
                    pending.cancel()

            # Close evolution tracer
            if self.evolution_tracer:
                self.evolution_tracer.close()
                logger.info("Evolution tracer closed")

        # Get the best program
        best_program = None
        if self.database.best_program_id:
            best_program = self.database.get(self.database.best_program_id)
            logger.info(f"Using tracked best program: {self.database.best_program_id}")

        if best_program is None:
            best_program = self.database.get_best_program()
            logger.info("Using calculated best program (tracked program not found)")

        if best_program:
            if (
                hasattr(self, "parallel_controller")
                and self.parallel_controller
                and self.parallel_controller.early_stopping_triggered
            ):
                logger.info(
                    f"🛑 Evolution complete via early stopping. Best program has metrics: "
                    f"{format_metrics_safe(best_program.metrics)}"
                )
            else:
                logger.info(
                    f"Evolution complete. Best program has metrics: "
                    f"{format_metrics_safe(best_program.metrics)}"
                )
            self._save_best_program(best_program)
            return best_program
        else:
            logger.warning("No valid programs found during evolution")
            return None

    def _log_iteration(
        self,
        iteration: int,
        parent: Program,
        child: Program,
        elapsed_time: float,
    ) -> None:
        """
        Log iteration progress

        Args:
            iteration: Iteration number
            parent: Parent program
            child: Child program
            elapsed_time: Elapsed time in seconds
        """
        # Calculate improvement using safe formatting
        improvement_str = format_improvement_safe(parent.metrics, child.metrics)

        logger.info(
            f"Iteration {iteration+1}: Child {child.id} from parent {parent.id} "
            f"in {elapsed_time:.2f}s. Metrics: "
            f"{format_metrics_safe(child.metrics)} "
            f"(Δ: {improvement_str})"
        )

    def _save_checkpoint(self, iteration: int) -> None:
        """
        Save a checkpoint

        Args:
            iteration: Current iteration number
        """
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Create specific checkpoint directory
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration}")
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save the database
        self.database.save(checkpoint_path, iteration)

        # Save the best program found so far
        best_program = None
        if self.database.best_program_id:
            best_program = self.database.get(self.database.best_program_id)
        else:
            best_program = self.database.get_best_program()

        if best_program:
            # Save the best program at this checkpoint
            best_program_path = os.path.join(checkpoint_path, f"best_program{self.file_extension}")
            with open(best_program_path, "w") as f:
                f.write(best_program.code)

            # Save metrics
            best_program_info_path = os.path.join(checkpoint_path, "best_program_info.json")
            with open(best_program_info_path, "w") as f:
                import json

                json.dump(
                    {
                        "id": best_program.id,
                        "generation": best_program.generation,
                        "iteration": best_program.iteration_found,
                        "current_iteration": iteration,
                        "metrics": best_program.metrics,
                        "language": best_program.language,
                        "timestamp": best_program.timestamp,
                        "saved_at": time.time(),
                    },
                    f,
                    indent=2,
                )

            logger.info(
                f"Saved best program at checkpoint {iteration} with metrics: "
                f"{format_metrics_safe(best_program.metrics)}"
            )

        logger.info(f"Saved checkpoint at iteration {iteration} to {checkpoint_path}")

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load state from a checkpoint directory"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint directory {checkpoint_path} not found")

        logger.info(f"Loading checkpoint from {checkpoint_path}")
        self.database.load(checkpoint_path)
        logger.info(f"Checkpoint loaded successfully (iteration {self.database.last_iteration})")

    async def _run_meta_analysis(self, iteration: int) -> None:
        """Periodically summarize recent research notes into agenda and findings.

        This is part of the AetherState R&D plan (Phase 4/5). It collects the
        most recent research notes, asks the LLM to update the cumulative
        findings and propose a short-term agenda, and writes them to disk so
        they can be injected into subsequent prompts.
        """
        # Derive the task directory from the initial program path so this works
        # regardless of where the openevolve package is installed.
        initial_path = Path(self.initial_program_path).resolve()
        aetherstate_dir = initial_path.parent
        if not aetherstate_dir.is_dir():
            return

        agenda_path = aetherstate_dir / "AGENDA.md"
        findings_path = aetherstate_dir / "FINDINGS.md"

        # Only run meta-analysis for the AetherState task.
        if "tasks/aetherstate" not in self.initial_program_path and not agenda_path.exists():
            return

        # Collect recent successful programs with research notes.
        recent_programs = sorted(
            self.database.programs.values(),
            key=lambda p: p.iteration_found,
            reverse=True,
        )[:20]

        notes = []
        for prog in recent_programs:
            note = prog.metadata.get("research_note", "")
            if note and note.strip():
                notes.append(
                    f"Program {prog.id[:8]} (iteration {prog.iteration_found}, "
                    f"fitness {prog.metrics.get('fitness', 0):.4f}):\n{note.strip()}"
                )

        if not notes:
            return

        prev_agenda = agenda_path.read_text(encoding="utf-8") if agenda_path.exists() else ""
        prev_findings = findings_path.read_text(encoding="utf-8") if findings_path.exists() else ""

        prompt = (
            "You are the principal investigator for an automated chess-AI research loop.\n\n"
            "Review the previous agenda and findings, then the recent research notes, "
            "and produce updated scientific direction.\n\n"
            "=== Previous Research Agenda ===\n" + prev_agenda + "\n\n"
            "=== Cumulative Findings ===\n" + prev_findings + "\n\n"
            "=== Recent Research Notes ===\n" + "\n\n".join(notes) + "\n\n"
            "Return a JSON object with exactly two keys: 'agenda' and 'findings'.\n"
            "- 'agenda' should be a concise markdown list (max 5 items) of concrete "
            "hypotheses to explore in the next block of iterations.\n"
            "- 'findings' should be a concise markdown summary of what worked, what "
            "failed, and open questions. Consolidate older findings to stay under "
            "800 words total.\n"
        )

        try:
            response = await self.llm_ensemble.generate_with_context(
                system_message=(
                    "You are a scientific research director. Produce only valid JSON "
                    "with keys 'agenda' and 'findings'. Keep the total response under "
                    "1000 words."
                ),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
        except Exception as e:
            logger.warning(f"Meta-analysis LLM call failed at iteration {iteration}: {e}")
            return

        if not response:
            logger.warning(f"Meta-analysis LLM returned empty response at iteration {iteration}")
            return

        try:
            import json

            text = response.strip()
            # Strip markdown code fences if the LLM wrapped the JSON.
            if text.startswith("```"):
                text = text.split("```", 2)[-1].strip()
            if text.startswith("json"):
                text = text[4:].strip()
            parsed = json.loads(text)
            agenda = parsed.get("agenda", "")
            findings = parsed.get("findings", "")
        except Exception as e:
            logger.warning(f"Could not parse meta-analysis JSON at iteration {iteration}: {e}")
            return

        # Atomic write to avoid readers seeing a half-written file.
        tmp_agenda = aetherstate_dir / "AGENDA.md.tmp"
        tmp_findings = aetherstate_dir / "FINDINGS.md.tmp"
        try:
            tmp_agenda.write_text(str(agenda), encoding="utf-8")
            tmp_findings.write_text(str(findings), encoding="utf-8")
            tmp_agenda.replace(agenda_path)
            tmp_findings.replace(findings_path)
            logger.info(f"Updated research agenda and findings at iteration {iteration}")
        except Exception as e:
            logger.warning(f"Failed to write meta-analysis files at iteration {iteration}: {e}")

    async def _run_evolution_with_checkpoints(
        self, start_iteration: int, max_iterations: int, target_score: Optional[float]
    ) -> None:
        """Run evolution with checkpoint saving support"""
        logger.info(f"Using island-based evolution with {self.config.database.num_islands} islands")
        self.database.log_island_status()

        # Keep a reference to any in-flight meta-analysis so it is not garbage
        # collected mid-flight.  Running it as a background task keeps the main
        # event loop free to reap completed worker results.
        self._pending_meta_analysis: Optional[asyncio.Task] = None

        async def _checkpoint_and_meta_analysis(iteration: int) -> None:
            self._save_checkpoint(iteration)
            if self._pending_meta_analysis is not None:
                # Wait for a previous meta-analysis to finish before starting a
                # new one, but do not block checkpointing on it.
                try:
                    await self._pending_meta_analysis
                except Exception:
                    pass
            self._pending_meta_analysis = asyncio.create_task(self._run_meta_analysis(iteration))

        # Run the evolution process with checkpoint callback
        await self.parallel_controller.run_evolution(
            start_iteration, max_iterations, target_score, checkpoint_callback=_checkpoint_and_meta_analysis
        )

        # Check if shutdown or early stopping was triggered
        if self.parallel_controller.shutdown_event.is_set():
            logger.info("Evolution stopped due to shutdown request")
            return
        elif self.parallel_controller.early_stopping_triggered:
            logger.info("Evolution stopped due to early stopping - saving final checkpoint")
            # Continue to save final checkpoint for early stopping

        # Save final checkpoint if needed
        # Note: start_iteration here is the evolution start (1 for fresh start, not 0)
        # max_iterations is the number of evolution iterations to run
        final_iteration = start_iteration + max_iterations - 1
        if final_iteration > 0 and final_iteration % self.config.checkpoint_interval == 0:
            self._save_checkpoint(final_iteration)

    def _save_best_program(self, program: Optional[Program] = None) -> None:
        """
        Save the best program

        Args:
            program: Best program (if None, uses the tracked best program)
        """
        # If no program is provided, use the tracked best program from the database
        if program is None:
            if self.database.best_program_id:
                program = self.database.get(self.database.best_program_id)
            else:
                # Fallback to calculating best program if no tracked best program
                program = self.database.get_best_program()

        if not program:
            logger.warning("No best program found to save")
            return

        best_dir = os.path.join(self.output_dir, "best")
        os.makedirs(best_dir, exist_ok=True)

        # Use the extension from the initial program file
        filename = f"best_program{self.file_extension}"
        code_path = os.path.join(best_dir, filename)

        with open(code_path, "w") as f:
            f.write(program.code)

        # Save complete program info including metrics
        info_path = os.path.join(best_dir, "best_program_info.json")
        with open(info_path, "w") as f:
            import json

            json.dump(
                {
                    "id": program.id,
                    "generation": program.generation,
                    "iteration": program.iteration_found,
                    "timestamp": program.timestamp,
                    "parent_id": program.parent_id,
                    "metrics": program.metrics,
                    "language": program.language,
                    "saved_at": time.time(),
                },
                f,
                indent=2,
            )

        logger.info(f"Saved best program to {code_path} with program info to {info_path}")
