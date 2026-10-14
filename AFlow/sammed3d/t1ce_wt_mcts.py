"""Typed MCTS primitives for the frozen BraTS T1ce/WT inference workflow.

The module deliberately separates label-free workflow execution from scoring.
It only composes precomputed robust-refiner logits; it does not train weights or
provide a test-set entrypoint.  The active geometry remains GT-centered oracle
geometry and must be described as such in every report.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from scipy import ndimage


ATOM_CACHE_SCHEMA = "aflow.sammed3d.t1ce_wt_robust_inference_atoms"
MCTS_SCHEMA = "aflow.sammed3d.t1ce_wt_inference_workflow_mcts"
VALIDATION_CACHE_INDEX_SHA256 = (
    "72fddef3c7a8755f0712b864efbb7328e2d8ae97313ac7ca7a0306d4b612ce44"
)
ROBUST_CHECKPOINT_SHA256 = (
    "011a7b0f4a9ee193883662f03fc6d8a54cef9fd3b4c08c2871cdddbdfbe73f6a",
    "c94f32c04cb89f51e1b83ae157bbce1d89371aa21b4267164e1ed645fa4d4cbc",
    "c5d2b48db2666f097cb0d86e78c20ea95d163c07f98401714d25d755e99aceb0",
)
ROBUST_SEEDS = (20260813, 20260814, 20260815)
LOCKED_SCAN_MACRO_DSC = 0.700418962602465
LOCKED_SUBJECT_MACRO_DSC = 0.6953719490880098
FORMAL_AGGREGATE_PARITY_TOLERANCE = 1e-5
FORMAL_PER_CASE_PARITY_TOLERANCE = 5e-4
ENSEMBLE_TTA_REPORT_SHA256 = "773fd8efc22221af4992c370ae73431787408fb693bc7af310ba298adc551388"
ENSEMBLE_REPORT_SHA256 = "0fefb2f9cdbabfe211233f04083abf243d8559044411e124e02924ba9e92cb03"
MODEL_SHAPE = (128, 128, 128)
REFLECTION_AXES: tuple[tuple[int, ...], ...] = (
    (),
    (-1,),
    (-2,),
    (-2, -1),
    (-3,),
    (-3, -1),
    (-3, -2),
    (-3, -2, -1),
)
VIEW_INDICES = {
    "identity": (0,),
    "single_flips": tuple(i for i, axes in enumerate(REFLECTION_AXES) if len(axes) <= 1),
    "all8": tuple(range(8)),
}
CONNECTIVITY_26 = ndimage.generate_binary_structure(3, 3)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_id(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def subject_id(case_id: str) -> str:
    """Return the BraTS subject id while keeping generic synthetic ids usable."""
    pieces = case_id.rsplit("-", 1)
    if len(pieces) == 2 and pieces[1].isdigit():
        return pieces[0]
    return case_id


@dataclass(frozen=True)
class WorkflowSpec:
    """A complete, canonical inference workflow.

    Quantized integers are used for searched floating point values so workflow
    ids are stable across Python/platform JSON formatting.
    """

    acquisition: str = "fixed"  # fixed | adaptive
    model_mask: int = 7
    view_policy: str = "all8"
    gate3_q: int = 0  # disagreement threshold / 100
    gate12_q: int = 0
    fusion: str = "logit_mean"  # logit_mean | probability_mean | logit_median
    threshold_q: int = -20  # raw-logit threshold / 20
    uncertainty_beta_q: int = 0  # beta / 10
    min_component_voxels: int = 500

    def __post_init__(self) -> None:
        if self.acquisition not in {"fixed", "adaptive"}:
            raise ValueError(f"unsupported acquisition: {self.acquisition}")
        if self.fusion not in {"logit_mean", "probability_mean", "logit_median"}:
            raise ValueError(f"unsupported fusion: {self.fusion}")
        if not -40 <= self.threshold_q <= 20:
            raise ValueError("threshold_q must encode a threshold in [-2, 1]")
        if not -20 <= self.uncertainty_beta_q <= 20:
            raise ValueError("uncertainty_beta_q must encode beta in [-2, 2]")
        if self.min_component_voxels < 0:
            raise ValueError("min_component_voxels cannot be negative")
        if self.acquisition == "fixed":
            if not 1 <= self.model_mask <= 7:
                raise ValueError("fixed model_mask must select at least one of three models")
            if self.view_policy not in VIEW_INDICES:
                raise ValueError(f"unsupported view policy: {self.view_policy}")
            if self.gate3_q != 0 or self.gate12_q != 0:
                raise ValueError("fixed acquisition cannot contain adaptive gates")
        else:
            if self.model_mask != 7 or self.view_policy != "all8":
                raise ValueError("adaptive acquisition is fixed to all three models and staged views")
            if not 0 <= self.gate3_q <= self.gate12_q <= 50:
                raise ValueError("adaptive gates must satisfy 0 <= gate3 <= gate12 <= 0.5")

    @property
    def threshold(self) -> float:
        return self.threshold_q / 20.0

    @property
    def uncertainty_beta(self) -> float:
        return self.uncertainty_beta_q / 10.0

    @property
    def workflow_id(self) -> str:
        return stable_id(self.canonical_dict())

    def canonical_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["schema_version"] = 1
        return value

    @classmethod
    def locked_reference(cls) -> "WorkflowSpec":
        return cls()


@dataclass(frozen=True)
class WorkflowOutput:
    fused_logits: np.ndarray
    voxel_uncertainty: np.ndarray | None
    forward_count: int
    stop_stage: int
    case_disagreement: float


def selected_atom_indices(model_mask: int, view_policy: str) -> tuple[int, ...]:
    indices: list[int] = []
    # Preserve the historical ensemble-TTA accumulation order exactly: the
    # evaluator iterates reflection first and model second.  This matters for a
    # strict float32 parity gate near the raw-logit threshold.
    for view in VIEW_INDICES[view_policy]:
        for model_slot in range(3):
            if model_mask & (1 << model_slot):
                indices.append(model_slot * 8 + view)
    if not indices:
        raise ValueError("workflow selected no inference atoms")
    return tuple(indices)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    out = np.empty_like(value, dtype=np.float32)
    positive = value >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    out[~positive] = exp_value / (1.0 + exp_value)
    return out


def _fuse(
    atoms: np.ndarray,
    indices: Sequence[int],
    kind: str,
    *,
    need_uncertainty: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    if not indices:
        raise ValueError("cannot fuse an empty atom set")
    if kind == "logit_median":
        fused = np.median(np.asarray(atoms[list(indices)], dtype=np.float32), axis=0).astype(
            np.float32, copy=False
        )
    elif kind == "probability_mean":
        total = np.zeros(atoms.shape[1:], dtype=np.float32)
        for index in indices:
            total += _sigmoid(atoms[index])
        probability = np.clip(total / len(indices), 1e-6, 1.0 - 1e-6)
        fused = np.log(probability / (1.0 - probability)).astype(np.float32, copy=False)
    else:
        total = np.zeros(atoms.shape[1:], dtype=np.float32)
        for index in indices:
            total += np.asarray(atoms[index], dtype=np.float32)
        fused = total / len(indices)

    uncertainty: np.ndarray | None = None
    if need_uncertainty and len(indices) > 1:
        mean = np.zeros(atoms.shape[1:], dtype=np.float32)
        mean_square = np.zeros_like(mean)
        for index in indices:
            probability = _sigmoid(atoms[index])
            mean += probability
            mean_square += probability * probability
        mean /= len(indices)
        variance = np.maximum(mean_square / len(indices) - mean * mean, 0.0)
        uncertainty = np.sqrt(variance).astype(np.float32, copy=False)
    return fused.astype(np.float32, copy=False), uncertainty


def case_disagreement(atoms: np.ndarray, indices: Sequence[int]) -> float:
    """Prediction-only mean pair disagreement used by adaptive early stopping."""
    if len(indices) < 2:
        return 0.0
    probabilities = np.stack(
        [_sigmoid(atoms[index]).reshape(-1) for index in indices], axis=0
    )
    gram = probabilities @ probabilities.T
    diagonal = np.diag(gram)
    left, right = np.triu_indices(len(indices), k=1)
    agreements = (2.0 * gram[left, right] + 1e-6) / (
        diagonal[left] + diagonal[right] + 1e-6
    )
    return float(1.0 - np.mean(agreements, dtype=np.float64))


def acquisition_decision(
    atoms: np.ndarray,
    spec: WorkflowSpec,
) -> tuple[tuple[int, ...], int, float]:
    """Choose label-free inference atoms and return indices, stage, disagreement."""
    if spec.acquisition == "fixed":
        indices = selected_atom_indices(spec.model_mask, spec.view_policy)
        return indices, len(indices), 0.0
    stage3 = selected_atom_indices(7, "identity")
    disagreement3 = case_disagreement(atoms, stage3)
    if disagreement3 <= spec.gate3_q / 100.0:
        return stage3, 3, disagreement3
    stage12 = selected_atom_indices(7, "single_flips")
    disagreement12 = case_disagreement(atoms, stage12)
    if disagreement12 <= spec.gate12_q / 100.0:
        return stage12, 12, disagreement12
    return selected_atom_indices(7, "all8"), 24, disagreement12


def run_workflow(atoms: np.ndarray, spec: WorkflowSpec) -> WorkflowOutput:
    """Execute one label-free workflow on one case's 24 canonical logit atoms."""
    if atoms.ndim < 2 or atoms.shape[0] != 24:
        raise ValueError("atoms must have shape [24, ...spatial dimensions]")
    indices, stage, disagreement = acquisition_decision(atoms, spec)
    fused, uncertainty = _fuse(
        atoms,
        indices,
        spec.fusion,
        need_uncertainty=spec.uncertainty_beta_q != 0,
    )
    return WorkflowOutput(fused, uncertainty, len(indices), stage, disagreement)


def workflow_mask(output: WorkflowOutput, spec: WorkflowSpec) -> np.ndarray:
    cutoff: float | np.ndarray = spec.threshold
    if spec.uncertainty_beta_q:
        if output.voxel_uncertainty is None:
            raise ValueError("nonzero uncertainty beta requires multiple inference atoms")
        cutoff = cutoff + spec.uncertainty_beta * output.voxel_uncertainty
    prediction = output.fused_logits > cutoff
    if spec.min_component_voxels == 0:
        return prediction
    components, count = ndimage.label(prediction, structure=CONNECTIVITY_26)
    if count == 0:
        return prediction
    sizes = np.bincount(components.reshape(-1))
    keep = sizes >= spec.min_component_voxels
    keep[0] = False
    return keep[components]


def binary_dice(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    intersection = int(np.count_nonzero(prediction & target))
    denominator = int(np.count_nonzero(prediction)) + int(np.count_nonzero(target))
    return 2.0 * intersection / denominator if denominator else 1.0


def aggregate_scores(case_ids: Sequence[str], values: Sequence[float]) -> dict[str, Any]:
    if len(case_ids) != len(values) or not values:
        raise ValueError("case ids and non-empty score values must have equal length")
    grouped: dict[str, list[float]] = {}
    for case, value in zip(case_ids, values, strict=True):
        grouped.setdefault(subject_id(case), []).append(float(value))
    subject_values = {key: float(np.mean(item)) for key, item in grouped.items()}
    return {
        "scan_macro_dsc": float(np.mean(values)),
        "subject_macro_dsc": float(np.mean(list(subject_values.values()))),
        "subjects": len(subject_values),
        "subject_values": subject_values,
    }


class _ByteLRU:
    def __init__(self, maximum_bytes: int) -> None:
        self.maximum_bytes = max(0, int(maximum_bytes))
        self.values: OrderedDict[tuple[str, int], WorkflowOutput] = OrderedDict()
        self.current_bytes = 0

    @staticmethod
    def _bytes(value: WorkflowOutput) -> int:
        total = value.fused_logits.nbytes
        if value.voxel_uncertainty is not None:
            total += value.voxel_uncertainty.nbytes
        return total

    def get(self, key: tuple[str, int]) -> WorkflowOutput | None:
        value = self.values.pop(key, None)
        if value is not None:
            self.values[key] = value
        return value

    def put(self, key: tuple[str, int], value: WorkflowOutput) -> None:
        if self.maximum_bytes <= 0:
            return
        size = self._bytes(value)
        if size > self.maximum_bytes:
            return
        old = self.values.pop(key, None)
        if old is not None:
            self.current_bytes -= self._bytes(old)
        while self.values and self.current_bytes + size > self.maximum_bytes:
            _, removed = self.values.popitem(last=False)
            self.current_bytes -= self._bytes(removed)
        self.values[key] = value
        self.current_bytes += size


class WorkflowEvaluator:
    """Score workflows while keeping labels out of ``run_workflow``."""

    def __init__(
        self,
        atoms: np.ndarray,
        labels: np.ndarray,
        case_ids: Sequence[str],
        *,
        memory_cache_bytes: int = 0,
    ) -> None:
        if atoms.shape[0] != labels.shape[0] or atoms.shape[0] != len(case_ids):
            raise ValueError("atom, label, and case counts differ")
        if atoms.shape[1] != 24:
            raise ValueError("atom cache must contain three models x eight reflections")
        if tuple(atoms.shape[2:]) != tuple(labels.shape[1:]):
            raise ValueError("atom and label spatial shapes differ")
        self.atoms = atoms
        self.labels = labels
        self.case_ids = list(case_ids)
        self.inference_cache = _ByteLRU(memory_cache_bytes)
        self.disagreement_cache: dict[tuple[int, int], float] = {}
        self.evaluation_cache: dict[tuple[str, tuple[int, ...]], dict[str, Any]] = {}

    def _decision(
        self,
        case_index: int,
        spec: WorkflowSpec,
    ) -> tuple[tuple[int, ...], int, float]:
        if spec.acquisition == "fixed":
            indices = selected_atom_indices(spec.model_mask, spec.view_policy)
            return indices, len(indices), 0.0

        stage3 = selected_atom_indices(7, "identity")
        disagreement3 = self.disagreement_cache.get((3, case_index))
        if disagreement3 is None:
            disagreement3 = case_disagreement(self.atoms[case_index], stage3)
            self.disagreement_cache[(3, case_index)] = disagreement3
        if disagreement3 <= spec.gate3_q / 100.0:
            return stage3, 3, disagreement3

        stage12 = selected_atom_indices(7, "single_flips")
        disagreement12 = self.disagreement_cache.get((12, case_index))
        if disagreement12 is None:
            disagreement12 = case_disagreement(self.atoms[case_index], stage12)
            self.disagreement_cache[(12, case_index)] = disagreement12
        if disagreement12 <= spec.gate12_q / 100.0:
            return stage12, 12, disagreement12
        return selected_atom_indices(7, "all8"), 24, disagreement12

    def _output(self, case_index: int, spec: WorkflowSpec) -> WorkflowOutput:
        indices, stage, disagreement = self._decision(case_index, spec)
        inference_key = stable_id(
            {
                "indices": indices,
                "fusion": spec.fusion,
                "need_uncertainty": spec.uncertainty_beta_q != 0,
            }
        )
        key = (inference_key, int(case_index))
        value = self.inference_cache.get(key)
        if value is None:
            fused, uncertainty = _fuse(
                self.atoms[case_index],
                indices,
                spec.fusion,
                need_uncertainty=spec.uncertainty_beta_q != 0,
            )
            value = WorkflowOutput(fused, uncertainty, len(indices), stage, disagreement)
            self.inference_cache.put(key, value)
        # Fused tensors are reusable across fixed/adaptive policies that select
        # the same atoms, but stop metadata belongs to the current decision.
        return WorkflowOutput(
            value.fused_logits,
            value.voxel_uncertainty,
            len(indices),
            stage,
            disagreement,
        )

    def evaluate(self, spec: WorkflowSpec, indices: Iterable[int]) -> dict[str, Any]:
        slots = tuple(sorted(int(value) for value in indices))
        if not slots:
            raise ValueError("cannot evaluate an empty case subset")
        cache_key = (spec.workflow_id, slots)
        cached = self.evaluation_cache.get(cache_key)
        if cached is not None:
            return cached
        values: list[float] = []
        forwards: list[int] = []
        stop_stages: list[int] = []
        disagreements: list[float] = []
        selected_ids: list[str] = []
        for slot in slots:
            output = self._output(slot, spec)
            prediction = workflow_mask(output, spec)
            values.append(binary_dice(prediction, self.labels[slot] > 0))
            forwards.append(output.forward_count)
            stop_stages.append(output.stop_stage)
            disagreements.append(output.case_disagreement)
            selected_ids.append(self.case_ids[slot])
        aggregate = aggregate_scores(selected_ids, values)
        result = {
            **{key: value for key, value in aggregate.items() if key != "subject_values"},
            "subject_values": aggregate["subject_values"],
            "mean_forwards": float(np.mean(forwards)),
            "max_forwards": int(max(forwards)),
            "stop_stage_counts": {
                str(stage): int(sum(value == stage for value in stop_stages))
                for stage in sorted(set(stop_stages))
            },
            "case_ids": selected_ids,
            "case_dice": values,
            "forward_counts": forwards,
            "case_disagreement": disagreements,
        }
        self.evaluation_cache[cache_key] = result
        return result

    def reward(self, spec: WorkflowSpec, indices: Iterable[int]) -> dict[str, Any]:
        slots = tuple(sorted(int(value) for value in indices))
        candidate = self.evaluate(spec, slots)
        reference = self.evaluate(WorkflowSpec.locked_reference(), slots)
        common_subjects = sorted(set(candidate["subject_values"]) & set(reference["subject_values"]))
        deltas = np.asarray(
            [candidate["subject_values"][key] - reference["subject_values"][key] for key in common_subjects],
            dtype=np.float64,
        )
        paired_se = float(np.std(deltas, ddof=1) / math.sqrt(len(deltas))) if len(deltas) > 1 else 0.0
        accuracy = candidate["subject_macro_dsc"]
        compute_penalty = 0.002 * math.log1p(candidate["mean_forwards"]) / math.log(25.0)
        complexity_terms = sum(
            (
                spec.acquisition == "adaptive",
                spec.fusion != "logit_mean",
                spec.uncertainty_beta_q != 0,
                spec.min_component_voxels != 0,
            )
        )
        complexity_penalty = 0.0005 * complexity_terms / 4.0
        reward = accuracy - 0.25 * paired_se - compute_penalty - complexity_penalty
        return {
            "workflow_id": spec.workflow_id,
            "spec": spec.canonical_dict(),
            "reward": float(reward),
            "accuracy": float(accuracy),
            "paired_subject_delta_mean": float(np.mean(deltas)),
            "paired_subject_delta_se": paired_se,
            "compute_penalty": float(compute_penalty),
            "complexity_penalty": float(complexity_penalty),
            **{key: value for key, value in candidate.items() if key != "subject_values"},
        }


class RobustAtomCache:
    """Fail-closed loader for the development-only robust validation atoms."""

    def __init__(
        self,
        root: Path,
        *,
        strict_repository_bindings: bool = True,
        allow_proxy: bool = False,
    ) -> None:
        self.root = root.resolve()
        lowered = str(self.root).lower()
        if "sealed_test" in lowered or "test270" in lowered:
            raise ValueError("test artifacts are structurally forbidden for workflow search")
        self.index_path = self.root / "index.json"
        self.index = json.loads(self.index_path.read_text())
        if self.index.get("schema") != ATOM_CACHE_SCHEMA:
            raise ValueError("unexpected inference atom cache schema")
        if self.index.get("split") != "validation" or self.index.get("test_access") is not False:
            raise ValueError("MCTS requires a development-only validation atom cache")
        if self.index.get("complete") is not True or self.index.get("parity", {}).get("passed") is not True:
            raise ValueError("atom cache is incomplete or failed the locked-workflow parity gate")
        if strict_repository_bindings:
            if int(self.index.get("count", -1)) != 108:
                raise ValueError("active validation atom cache must contain exactly 108 scans")
            if self.index.get("validation_cache_index_sha256") != VALIDATION_CACHE_INDEX_SHA256:
                raise RuntimeError("validation cache index binding mismatch")
            checkpoint_hashes = tuple(item["sha256"] for item in self.index.get("checkpoints", []))
            if checkpoint_hashes != ROBUST_CHECKPOINT_SHA256:
                raise RuntimeError("robust checkpoint allowlist mismatch")
        expected_order = [
            {
                "atom_index": model_slot * 8 + view_slot,
                "model_slot": model_slot,
                "seed": ROBUST_SEEDS[model_slot],
                "checkpoint_sha256": ROBUST_CHECKPOINT_SHA256[model_slot],
                "reflection_axes": list(axes),
                "inverse_reflection_axes": list(axes),
            }
            for model_slot in range(3)
            for view_slot, axes in enumerate(REFLECTION_AXES)
        ]
        if self.index.get("atom_order") != expected_order:
            raise RuntimeError("inference atom order/transform binding mismatch")
        dtype = np.dtype(self.index["dtype"])
        shape = tuple(int(value) for value in self.index["shape"])
        expected_shape = (int(self.index["count"]), 24, *MODEL_SHAPE)
        if shape != expected_shape:
            raise ValueError("atom cache shape is not [count,3x8,128,128,128]")
        evidence_level = self.index.get("evidence_level")
        if not allow_proxy and (dtype != np.dtype("float32") or evidence_level != "exact"):
            raise ValueError("formal MCTS requires exact float32 atoms; compact atoms are proxy-only")
        if not allow_proxy and self.index.get("core_sha256") != sha256_file(Path(__file__).resolve()):
            raise RuntimeError("atom cache was built by a different MCTS workflow implementation")
        parity = self.index.get("parity", {})
        if parity.get("reference") != WorkflowSpec.locked_reference().canonical_dict():
            raise RuntimeError("locked workflow parity reference mismatch")
        aggregate_tolerance = float(
            parity.get("aggregate_absolute_tolerance", math.inf)
        )
        per_case_tolerance = float(
            parity.get("per_case_absolute_tolerance", math.inf)
        )
        if (
            parity.get("passed") is not True
            or not np.isfinite(aggregate_tolerance)
            or not np.isfinite(per_case_tolerance)
            or aggregate_tolerance < 0
            or per_case_tolerance < 0
            or (
                not allow_proxy
                and aggregate_tolerance > FORMAL_AGGREGATE_PARITY_TOLERANCE
            )
            or (
                not allow_proxy
                and per_case_tolerance > FORMAL_PER_CASE_PARITY_TOLERANCE
            )
            or abs(float(parity.get("actual_scan_macro_dsc", -1)) - LOCKED_SCAN_MACRO_DSC)
            > aggregate_tolerance
            or abs(
                float(parity.get("actual_subject_macro_dsc", -1))
                - LOCKED_SUBJECT_MACRO_DSC
            )
            > aggregate_tolerance
            or float(parity.get("maximum_per_case_absolute_dice_error", math.inf))
            > per_case_tolerance
        ):
            raise RuntimeError("locked workflow parity values/tolerance are invalid")
        evidence = self.index.get("historical_parity_evidence", {})
        if (
            evidence.get("ensemble_tta_report_sha256") != ENSEMBLE_TTA_REPORT_SHA256
            or evidence.get("ensemble_report_sha256") != ENSEMBLE_REPORT_SHA256
            or evidence.get("per_case_passed") is not True
        ):
            raise RuntimeError("historical per-case parity evidence is missing or mismatched")
        data_path = self.root / self.index["atoms_file"]
        if sha256_file(data_path) != self.index["atoms_sha256"]:
            raise RuntimeError("inference atom file hash mismatch")
        self.atoms = np.memmap(data_path, dtype=dtype, mode="r", shape=shape)
        self.case_ids = list(self.index["case_ids"])
        if len(self.case_ids) != shape[0]:
            raise RuntimeError("atom cache case id count mismatch")


@dataclass(frozen=True)
class SearchSpace:
    acquisition_actions: tuple[str, ...] = tuple(
        [f"fixed:{mask}:{policy}" for mask in range(1, 8) for policy in VIEW_INDICES]
        + [
            f"adaptive:{gate3}:{gate12}"
            for gate3 in (0, 5, 10, 20, 30)
            for gate12 in (5, 10, 20, 30, 40)
            if gate12 >= gate3
        ]
    )
    fusion_actions: tuple[str, ...] = ("logit_mean", "probability_mean", "logit_median")
    threshold_actions: tuple[int, ...] = (-20, 0, -10, -30, -15, -5, 5, 10)
    beta_actions: tuple[int, ...] = (0, 10, -10, 5, -5, 20, -20)
    component_actions: tuple[int, ...] = (500, 0, 100, 1000, 250, 750, 1500, 2000)

    def canonical_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchState:
    acquisition_action: str | None = None
    fusion: str | None = None
    threshold_q: int | None = None
    uncertainty_beta_q: int | None = None
    min_component_voxels: int | None = None

    @property
    def terminal(self) -> bool:
        return self.min_component_voxels is not None

    @property
    def state_id(self) -> str:
        return stable_id(asdict(self))

    def legal_actions(self, space: SearchSpace) -> tuple[Any, ...]:
        if self.acquisition_action is None:
            return space.acquisition_actions
        if self.fusion is None:
            pieces = self.acquisition_action.split(":")
            if pieces[0] == "fixed":
                atom_count = int(pieces[1]).bit_count() * len(VIEW_INDICES[pieces[2]])
                if atom_count == 1:
                    return ("logit_mean",)
            return space.fusion_actions
        if self.threshold_q is None:
            return space.threshold_actions
        if self.uncertainty_beta_q is None:
            action = self.acquisition_action
            assert action is not None
            single_atom = action.startswith("fixed:") and (
                int(action.split(":")[1]).bit_count() * len(VIEW_INDICES[action.split(":")[2]]) == 1
            )
            return (0,) if single_atom else space.beta_actions
        if self.min_component_voxels is None:
            return space.component_actions
        return ()

    def apply(self, action: Any, space: SearchSpace) -> "SearchState":
        if action not in self.legal_actions(space):
            raise ValueError(f"illegal or duplicate action {action!r} for state")
        values = asdict(self)
        for name in (
            "acquisition_action",
            "fusion",
            "threshold_q",
            "uncertainty_beta_q",
            "min_component_voxels",
        ):
            if values[name] is None:
                values[name] = action
                return SearchState(**values)
        raise ValueError("terminal state has no legal action")

    def to_spec(self) -> WorkflowSpec:
        if not self.terminal:
            raise ValueError("partial search state cannot become a workflow")
        assert self.acquisition_action is not None
        assert self.fusion is not None
        assert self.threshold_q is not None
        assert self.uncertainty_beta_q is not None
        assert self.min_component_voxels is not None
        pieces = self.acquisition_action.split(":")
        if pieces[0] == "fixed":
            acquisition_values = {
                "acquisition": "fixed",
                "model_mask": int(pieces[1]),
                "view_policy": pieces[2],
                "gate3_q": 0,
                "gate12_q": 0,
            }
        else:
            acquisition_values = {
                "acquisition": "adaptive",
                "model_mask": 7,
                "view_policy": "all8",
                "gate3_q": int(pieces[1]),
                "gate12_q": int(pieces[2]),
            }
        return WorkflowSpec(
            **acquisition_values,
            fusion=self.fusion,
            threshold_q=self.threshold_q,
            uncertainty_beta_q=self.uncertainty_beta_q,
            min_component_voxels=self.min_component_voxels,
        )


@dataclass
class SearchNode:
    state: SearchState
    parent_id: str | None
    action: Any = None
    visits: int = 0
    value_sum: float = 0.0
    children: dict[str, str] = field(default_factory=dict)

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


def uct_score(parent_visits: int, child: SearchNode, exploration: float) -> float:
    if child.visits == 0:
        return math.inf
    return child.mean_value + exploration * math.sqrt(
        math.log(max(1, parent_visits)) / child.visits
    )


class MCTSSearch:
    """Standard UCT selection, stochastic rollout, and ancestor backup."""

    def __init__(
        self,
        evaluator: Callable[[WorkflowSpec], dict[str, Any]],
        *,
        space: SearchSpace | None = None,
        seed: int = 20260813,
        exploration: float = 0.02,
        widening_k: float = 2.0,
        widening_alpha: float = 0.5,
        context_id: str = "",
    ) -> None:
        self.evaluator = evaluator
        self.space = space or SearchSpace()
        self.seed = int(seed)
        self.exploration = float(exploration)
        self.widening_k = float(widening_k)
        self.widening_alpha = float(widening_alpha)
        self.context_id = str(context_id)
        self.rng = np.random.default_rng(self.seed)
        root = SearchNode(SearchState(), None)
        self.root_id = root.state.state_id
        self.nodes: dict[str, SearchNode] = {self.root_id: root}
        self.evaluations: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []

    @staticmethod
    def _action_token(action: Any) -> str:
        return canonical_json(action)

    def _expandable_count(self, node: SearchNode, action_count: int) -> int:
        return min(
            action_count,
            max(1, int(self.widening_k * (node.visits + 1) ** self.widening_alpha)),
        )

    def _choose_unexpanded(self, node: SearchNode, actions: Sequence[Any]) -> Any:
        candidates = [action for action in actions if self._action_token(action) not in node.children]
        return candidates[int(self.rng.integers(0, len(candidates)))]

    def _select_child(self, node: SearchNode) -> tuple[SearchNode, list[dict[str, Any]]]:
        ranked: list[tuple[float, str, SearchNode]] = []
        audit: list[dict[str, Any]] = []
        for token, child_id in node.children.items():
            child = self.nodes[child_id]
            exploration_term = (
                math.inf
                if child.visits == 0
                else self.exploration
                * math.sqrt(math.log(max(1, node.visits)) / child.visits)
            )
            score = uct_score(node.visits, child, self.exploration)
            ranked.append((score, token, child))
            audit.append(
                {
                    "action_token": token,
                    "child_state_id": child_id,
                    "N": child.visits,
                    "Q": child.mean_value,
                    "U": None if math.isinf(exploration_term) else exploration_term,
                    "UCT": None if math.isinf(score) else score,
                }
            )
        ranked.sort(key=lambda value: (-value[0], value[1]))
        audit.sort(key=lambda value: value["action_token"])
        return ranked[0][2], audit

    def _rollout(self, state: SearchState) -> tuple[SearchState, list[Any]]:
        actions_taken: list[Any] = []
        while not state.terminal:
            actions = state.legal_actions(self.space)
            action = actions[int(self.rng.integers(0, len(actions)))]
            actions_taken.append(action)
            state = state.apply(action, self.space)
        return state, actions_taken

    def simulate(self) -> dict[str, Any]:
        node = self.nodes[self.root_id]
        path = [node]
        selection_path: list[dict[str, Any]] = []
        expanded = False
        while not node.state.terminal:
            actions = node.state.legal_actions(self.space)
            expandable = self._expandable_count(node, len(actions))
            if len(node.children) < expandable:
                action = self._choose_unexpanded(node, actions)
                child_state = node.state.apply(action, self.space)
                child_id = child_state.state_id
                child = self.nodes.get(child_id)
                if child is None:
                    child = SearchNode(child_state, node.state.state_id, action=action)
                    self.nodes[child_id] = child
                node.children[self._action_token(action)] = child_id
                node = child
                path.append(node)
                selection_path.append(
                    {
                        "action": action,
                        "state_id": child_id,
                        "kind": "expand",
                        "legal_actions": len(actions),
                        "widening_cap": expandable,
                    }
                )
                expanded = True
                break
            node, child_statistics = self._select_child(node)
            path.append(node)
            selection_path.append(
                {
                    "action": node.action,
                    "state_id": node.state.state_id,
                    "kind": "select",
                    "legal_actions": len(actions),
                    "widening_cap": expandable,
                    "child_statistics": child_statistics,
                }
            )

        if node.state.terminal:
            terminal_state, rollout_actions = node.state, []
        else:
            terminal_state, rollout_actions = self._rollout(node.state)
        spec = terminal_state.to_spec()
        cache_hit = spec.workflow_id in self.evaluations
        if cache_hit:
            evaluation = self.evaluations[spec.workflow_id]
        else:
            evaluation = self.evaluator(spec)
            if evaluation.get("workflow_id") != spec.workflow_id:
                raise RuntimeError("evaluator returned a mismatched workflow id")
            self.evaluations[spec.workflow_id] = evaluation
        reward = float(evaluation["reward"])
        for ancestor in path:
            ancestor.visits += 1
            ancestor.value_sum += reward
        event = {
            "iteration": len(self.events),
            "expanded": expanded,
            "selection_path": selection_path,
            "rollout_actions": rollout_actions,
            "rollout_workflow_id": spec.workflow_id,
            "cache_hit": cache_hit,
            "reward": reward,
            "root_visits_after_backup": self.nodes[self.root_id].visits,
        }
        self.events.append(event)
        return event

    def run(self, simulations: int) -> None:
        if simulations < 1:
            raise ValueError("simulations must be positive")
        for _ in range(simulations):
            self.simulate()

    def ranked_evaluations(self) -> list[dict[str, Any]]:
        return sorted(
            self.evaluations.values(),
            key=lambda item: (-float(item["reward"]), float(item.get("mean_forwards", math.inf)), item["workflow_id"]),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": MCTS_SCHEMA,
            "schema_version": 1,
            "seed": self.seed,
            "exploration": self.exploration,
            "widening_k": self.widening_k,
            "widening_alpha": self.widening_alpha,
            "context_id": self.context_id,
            "space": self.space.canonical_dict(),
            "rng_state": self.rng.bit_generator.state,
            "root_id": self.root_id,
            "nodes": {
                key: {
                    "state": asdict(value.state),
                    "parent_id": value.parent_id,
                    "action": value.action,
                    "visits": value.visits,
                    "value_sum": value.value_sum,
                    "children": value.children,
                }
                for key, value in self.nodes.items()
            },
            "evaluations": self.evaluations,
            "events": self.events,
        }

    @classmethod
    def from_state_dict(
        cls,
        value: dict[str, Any],
        evaluator: Callable[[WorkflowSpec], dict[str, Any]],
        *,
        expected_context_id: str = "",
    ) -> "MCTSSearch":
        if value.get("schema") != MCTS_SCHEMA:
            raise ValueError("unexpected MCTS state schema")
        if value.get("context_id", "") != expected_context_id:
            raise RuntimeError("MCTS resume context binding mismatch")
        space = SearchSpace(**{key: tuple(item) for key, item in value["space"].items()})
        search = cls(
            evaluator,
            space=space,
            seed=int(value["seed"]),
            exploration=float(value["exploration"]),
            widening_k=float(value["widening_k"]),
            widening_alpha=float(value["widening_alpha"]),
            context_id=expected_context_id,
        )
        search.root_id = value["root_id"]
        search.nodes = {
            key: SearchNode(
                state=SearchState(**item["state"]),
                parent_id=item["parent_id"],
                action=item["action"],
                visits=int(item["visits"]),
                value_sum=float(item["value_sum"]),
                children=dict(item["children"]),
            )
            for key, item in value["nodes"].items()
        }
        search.evaluations = dict(value["evaluations"])
        search.events = list(value["events"])
        search.rng.bit_generator.state = value["rng_state"]
        return search


def build_subject_folds(
    case_ids: Sequence[str],
    *,
    seed: int = 20260813,
    folds: int = 3,
    allowed_indices: Iterable[int] | None = None,
) -> list[list[int]]:
    """Deterministic greedy fold allocation with zero subject overlap."""
    if folds < 2:
        raise ValueError("at least two folds are required")
    slots = list(range(len(case_ids))) if allowed_indices is None else sorted(allowed_indices)
    grouped: dict[str, list[int]] = {}
    for slot in slots:
        grouped.setdefault(subject_id(case_ids[slot]), []).append(slot)
    if len(grouped) < folds:
        raise ValueError("fewer subjects than requested folds")

    def tie_hash(key: str) -> str:
        return hashlib.sha256(f"{seed}|{key}".encode("utf-8")).hexdigest()

    ordered = sorted(grouped.items(), key=lambda item: (-len(item[1]), tie_hash(item[0])))
    result: list[list[int]] = [[] for _ in range(folds)]
    subject_counts = [0] * folds
    for _, subject_slots in ordered:
        target = min(
            range(folds),
            key=lambda index: (len(result[index]), subject_counts[index], index),
        )
        result[target].extend(subject_slots)
        subject_counts[target] += 1
    return [sorted(item) for item in result]


def nested_roles(
    case_ids: Sequence[str],
    outer_fold: int,
    *,
    seed: int = 20260813,
) -> dict[str, list[int]]:
    outer = build_subject_folds(case_ids, seed=seed, folds=3)
    if not 0 <= outer_fold < len(outer):
        raise ValueError("outer_fold must be 0, 1, or 2")
    holdout = outer[outer_fold]
    remaining = sorted(set(range(len(case_ids))) - set(holdout))
    inner = build_subject_folds(
        case_ids,
        seed=seed + 1009 * (outer_fold + 1),
        folds=2,
        allowed_indices=remaining,
    )
    roles = {"mcts_search": inner[0], "inner_selector": inner[1], "outer_holdout": holdout}
    subject_sets = {
        role: {subject_id(case_ids[index]) for index in indices}
        for role, indices in roles.items()
    }
    if any(
        subject_sets[left] & subject_sets[right]
        for left, right in (("mcts_search", "inner_selector"), ("mcts_search", "outer_holdout"), ("inner_selector", "outer_holdout"))
    ):
        raise RuntimeError("nested validation roles overlap by subject")
    return roles
