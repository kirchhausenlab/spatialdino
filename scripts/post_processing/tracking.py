from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from spatialdino.inference.output_layout import (
    TRACKS_FILENAME,
    discover_inference_timepoints,
)
from spatialdino.tracking.metrics import (
    LabelGeometry,
    PipelineError,
    anisotropic_distance,
    build_axis_maps,
    choose_feature_batch_size,
    compute_alignment_overlap,
    compute_label_mean_intensities,
    extract_label_geometries,
    load_feature_chunk_internal_yxz,
    read_tiff_volume,
    rowwise_correlation,
    sample_feature_chunk_at_internal_coords,
)

DEFAULT_N_PATCH_FEATURES = 384
CSV_NAN = "NaN"


@dataclass
class TimepointPaths:
    name: str
    raw_path: Path
    segmentation_path: Path
    lr_feats_path: Path


@dataclass
class PreparedTimepoint:
    index: int
    paths: TimepointPaths
    shape_yxz: tuple[int, int, int]
    raw_z_size: int
    geometries: dict[int, LabelGeometry]
    label_ids: np.ndarray
    centroids: np.ndarray
    volumes: np.ndarray
    amplitudes: np.ndarray


@dataclass
class CandidateVote:
    candidate_label: int
    wins: int = 0
    features: list[int] = field(default_factory=list)
    distance: float | None = None
    dice: float | None = None


@dataclass
class RefSummary:
    ref_label: int
    candidates: list[CandidateVote]


@dataclass
class AssignmentRecord:
    ref_label: int
    candidate_label: int
    method: str
    stage: int
    distance: float | None = None
    wins: int | None = None
    vote_threshold: int | None = None
    dice: float | None = None
    corr: float | None = None
    mse: float | None = None


@dataclass
class SegmentDiagnostics:
    stage: int
    assignment_method: str
    dice: float | None
    corr: float | None
    mse: float | None
    feat_votes: int | None
    vote_threshold: int | None
    anisotropic_distance: float | None


@dataclass
class PairResult:
    t_ref: int
    t_cand: int
    initial_summary: list[RefSummary]
    summary_current: list[RefSummary]
    summary_history: list[list[RefSummary]]
    assignments: list[AssignmentRecord]
    final_assignment: dict[int, int]
    summary_distance_prefilter: list[dict[str, float]]


@dataclass
class TrackPoint:
    timepoint: int
    label_id: int
    centroid: tuple[float, float, float]
    volume: int
    amplitude: float
    incoming: SegmentDiagnostics | None = None


@dataclass
class Track:
    track_id: int
    points: list[TrackPoint]
    start_time: int
    length: int


@dataclass
class RefCandidateMetrics:
    ref_label: int
    ref_centroid: np.ndarray
    candidate_ids: np.ndarray
    candidate_centroids: np.ndarray
    distances: np.ndarray
    dice: np.ndarray
    overlap_counts: np.ndarray
    corr: np.ndarray
    mse: np.ndarray


@dataclass(frozen=True)
class TrackingParams:
    max_distance_xy: float
    max_distance_z: float
    z_distance_weight: float
    min_distance_to_remove_cand: float
    vote_thresholds: tuple[int, ...] | None
    dice_threshold: float
    corr_threshold: float
    invert_z: bool
    save_extended_results: bool = False
    ignore_features: bool = False
    disable_centroid_fallback: bool = False
    aggressive_feature_matching: bool = False
    min_feature_votes: int = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track segmented objects across timepoints.")
    parser.add_argument("--input-path", required=True, help="Inference output folder containing lr_feats/ and raw/.")
    parser.add_argument(
        "--segmentation-path",
        required=True,
        help="Folder containing one segmentation mask per timepoint, named <timepoint>.tif.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Folder where the tracking CSV will be written. Defaults to the input folder.",
    )
    parser.add_argument(
        "--output-filename",
        default=TRACKS_FILENAME,
        help=f"CSV filename to write inside the output folder. Defaults to {TRACKS_FILENAME}.",
    )
    parser.add_argument(
        "--max-distance-xy",
        type=float,
        default=20.0,
        help="Maximum candidate centroid displacement along X and Y, in voxels.",
    )
    parser.add_argument(
        "--max-distance-z",
        type=float,
        default=10.0,
        help="Maximum candidate centroid displacement along Z, in voxels.",
    )
    parser.add_argument(
        "--z-distance-weight",
        type=float,
        default=2.5,
        help="Anisotropy factor applied to Z in the centroid distance metric.",
    )
    parser.add_argument(
        "--min-distance-to-remove-cand",
        type=float,
        default=3.0,
        help="Immediate-assignment threshold from the original MATLAB logic.",
    )
    parser.add_argument(
        "--vote-thresholds",
        type=str,
        default="320,300,280,260",
        help="Comma-separated minimum vote counts, e.g. '320,300,280,260'. Leave blank to use defaults.",
    )
    parser.add_argument(
        "--dice-threshold",
        type=float,
        default=0.5,
        help="Minimum centroid-aligned Dice overlap required for feature voting.",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.5,
        help="Minimum feature correlation required for a feature vote.",
    )
    parser.add_argument(
        "--save-extended-results",
        action="store_true",
        help="Append per-row tracking diagnostics to the output CSV.",
    )
    parser.add_argument(
        "--ignore-features",
        action="store_true",
        help="Skip feature voting and resolve links using distance stages only.",
    )
    parser.add_argument(
        "--disable-centroid-fallback",
        action="store_true",
        help="Skip the final centroid-only global-closest assignment stage.",
    )
    parser.add_argument(
        "--aggressive-feature-matching",
        action="store_true",
        help="Greedily assign remaining links using feature-vote evidence before centroid fallback.",
    )
    parser.add_argument(
        "--min-feature-votes",
        type=int,
        default=1,
        help="Minimum feature votes required for aggressive feature matching.",
    )
    invert_group = parser.add_mutually_exclusive_group()
    invert_group.add_argument(
        "--invert-z",
        dest="invert_z",
        action="store_true",
        help="Export z coordinates as z0 - z instead of keeping them as-is.",
    )
    invert_group.add_argument(
        "--no-invert-z",
        dest="invert_z",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(invert_z=False)
    return parser.parse_args()


def parse_thresholds(text: str | None) -> tuple[int, ...] | None:
    if text is None or text.strip() == "":
        return None
    thresholds: list[int] = []
    for token in text.split(","):
        value = token.strip()
        if not value:
            continue
        thresholds.append(int(value))
    return tuple(thresholds)


def normalize_output_filename(filename: str | None) -> str:
    value = TRACKS_FILENAME if filename is None else filename.strip()
    if not value:
        raise PipelineError("Output filename must not be empty.")
    path = Path(value)
    if path.name != value or value in {".", ".."} or "/" in value or "\\" in value:
        raise PipelineError("Output filename must be a file name, not a path.")
    if path.suffix == "":
        value = f"{value}.csv"
        path = Path(value)
    if path.suffix.lower() != ".csv":
        raise PipelineError("Output filename must end in .csv.")
    return value


def mask_path_for_timepoint(segmentation_path: Path, *, timepoint_name: str) -> Path:
    return segmentation_path / f"{timepoint_name}.tif"


def validate_timepoint(timepoint: Any, *, segmentation_path: Path) -> TimepointPaths:
    segmentation_mask_path = mask_path_for_timepoint(segmentation_path, timepoint_name=timepoint.name)
    if not segmentation_mask_path.is_file():
        raise FileNotFoundError(f"Missing segmentation mask for {timepoint.name}: {segmentation_mask_path.name}.")
    return TimepointPaths(
        name=timepoint.name,
        raw_path=timepoint.raw_path,
        segmentation_path=segmentation_mask_path,
        lr_feats_path=timepoint.lr_path,
    )


def discover_experiment(input_path: Path, *, segmentation_path: Path) -> list[TimepointPaths]:
    discovered: list[TimepointPaths] = []
    for timepoint in discover_inference_timepoints(input_path):
        discovered.append(validate_timepoint(timepoint, segmentation_path=segmentation_path))
    return discovered


def prepare_timepoint(index: int, paths: TimepointPaths) -> PreparedTimepoint:
    raw_yxz = read_tiff_volume(paths.raw_path)
    segmentation_yxz = read_tiff_volume(paths.segmentation_path)
    if raw_yxz.shape != segmentation_yxz.shape:
        raise PipelineError(
            (
                f"Shape mismatch in {paths.name}: "
                f"{paths.raw_path.name}={raw_yxz.shape}, {paths.segmentation_path.name}={segmentation_yxz.shape}."
            )
        )

    geometries = extract_label_geometries(segmentation_yxz)
    label_ids = np.array(sorted(geometries.keys()), dtype=np.int64)
    if label_ids.size == 0:
        centroids = np.empty((0, 3), dtype=np.float64)
        volumes = np.empty((0,), dtype=np.int64)
        amplitudes = np.empty((0,), dtype=np.float32)
    else:
        centroids = np.vstack([geometries[int(label_id)].centroid for label_id in label_ids])
        volumes = np.array([geometries[int(label_id)].volume for label_id in label_ids], dtype=np.int64)
        amplitude_labels, amplitudes = compute_label_mean_intensities(segmentation_yxz, raw_yxz)
        if not np.array_equal(amplitude_labels, label_ids):
            raise PipelineError(f"Label intensity extraction mismatch in {paths.name}.")

    return PreparedTimepoint(
        index=index,
        paths=paths,
        shape_yxz=tuple(int(dim) for dim in segmentation_yxz.shape),
        raw_z_size=int(raw_yxz.shape[2]),
        geometries=geometries,
        label_ids=label_ids,
        centroids=centroids,
        volumes=volumes,
        amplitudes=amplitudes,
    )


def find_spatial_candidates(
    ref_tp: PreparedTimepoint,
    cand_tp: PreparedTimepoint,
    *,
    spatial_radius: tuple[float, float, float],
    zratio: float,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    if cand_tp.label_ids.size == 0:
        return {
            int(ref_id): (
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.float32),
            )
            for ref_id in ref_tp.label_ids.tolist()
        }

    radius = np.asarray(spatial_radius, dtype=np.float64)
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for ref_id in ref_tp.label_ids.tolist():
        ref_centroid = ref_tp.geometries[int(ref_id)].centroid
        in_box = np.all(
            np.abs(cand_tp.centroids - ref_centroid[None, :]) <= radius[None, :],
            axis=1,
        )
        cand_ids = cand_tp.label_ids[in_box]
        distances = np.array(
            [
                anisotropic_distance(ref_centroid, cand_tp.geometries[int(cand_id)].centroid, zratio)
                for cand_id in cand_ids.tolist()
            ],
            dtype=np.float32,
        )
        order = np.argsort(distances, kind="mergesort")
        out[int(ref_id)] = (
            cand_ids[order].astype(np.int64, copy=False),
            distances[order].astype(np.float32, copy=False),
        )
    return out


def compute_pair_metrics(
    ref_tp: PreparedTimepoint,
    cand_tp: PreparedTimepoint,
    *,
    n_features: int,
    spatial_radius: tuple[float, float, float],
    zratio: float,
    ignore_features: bool = False,
) -> dict[int, RefCandidateMetrics]:
    spatial = find_spatial_candidates(
        ref_tp,
        cand_tp,
        spatial_radius=spatial_radius,
        zratio=zratio,
    )

    metrics_by_ref: dict[int, RefCandidateMetrics] = {}
    overlap_cache: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for ref_id in ref_tp.label_ids.tolist():
        candidate_ids, distances = spatial[int(ref_id)]
        candidate_count = len(candidate_ids)
        dice = np.full((candidate_count,), np.nan, dtype=np.float32)
        overlap_counts = np.zeros((candidate_count,), dtype=np.int32)
        overlaps: list[tuple[np.ndarray, np.ndarray]] = []

        ref_geom = ref_tp.geometries[int(ref_id)]
        for index, cand_id in enumerate(candidate_ids.tolist()):
            cand_geom = cand_tp.geometries[int(cand_id)]
            dice_value, ref_coords, cand_coords, overlap_count = compute_alignment_overlap(ref_geom, cand_geom)
            dice[index] = np.float32(dice_value)
            overlap_counts[index] = int(overlap_count)
            overlaps.append((ref_coords, cand_coords))

        metrics_by_ref[int(ref_id)] = RefCandidateMetrics(
            ref_label=int(ref_id),
            ref_centroid=ref_geom.centroid.astype(np.float64, copy=True),
            candidate_ids=candidate_ids,
            candidate_centroids=np.vstack(
                [cand_tp.geometries[int(cand_id)].centroid for cand_id in candidate_ids.tolist()]
            ).astype(np.float64, copy=False)
            if candidate_count
            else np.empty((0, 3), dtype=np.float64),
            distances=distances.astype(np.float32, copy=False),
            dice=dice,
            overlap_counts=overlap_counts,
            corr=np.full((n_features, candidate_count), np.nan, dtype=np.float32),
            mse=np.full((n_features, candidate_count), np.nan, dtype=np.float32),
        )
        overlap_cache[int(ref_id)] = overlaps

    if ignore_features:
        return metrics_by_ref

    ref_lr_feats = np.load(ref_tp.paths.lr_feats_path, mmap_mode="r")
    cand_lr_feats = np.load(cand_tp.paths.lr_feats_path, mmap_mode="r")
    if ref_lr_feats.ndim != 4:
        raise PipelineError(f"{ref_tp.paths.lr_feats_path} must have shape [Z, Y, X, C].")
    if cand_lr_feats.ndim != 4:
        raise PipelineError(f"{cand_tp.paths.lr_feats_path} must have shape [Z, Y, X, C].")
    if ref_lr_feats.shape[-1] < n_features:
        raise PipelineError(
            f"{ref_tp.paths.lr_feats_path} only has {ref_lr_feats.shape[-1]} channels; tracking needs {n_features}."
        )
    if cand_lr_feats.shape[-1] < n_features:
        raise PipelineError(
            f"{cand_tp.paths.lr_feats_path} only has {cand_lr_feats.shape[-1]} channels; tracking needs {n_features}."
        )

    ref_input_shape_yxz = (
        int(ref_lr_feats.shape[1]),
        int(ref_lr_feats.shape[2]),
        int(ref_lr_feats.shape[0]),
    )
    cand_input_shape_yxz = (
        int(cand_lr_feats.shape[1]),
        int(cand_lr_feats.shape[2]),
        int(cand_lr_feats.shape[0]),
    )
    ref_axis_maps = build_axis_maps(ref_input_shape_yxz, ref_tp.shape_yxz)
    cand_axis_maps = build_axis_maps(cand_input_shape_yxz, cand_tp.shape_yxz)
    batch_size = choose_feature_batch_size(
        tuple(int(dim) for dim in ref_lr_feats.shape[:3]),
        tuple(int(dim) for dim in cand_lr_feats.shape[:3]),
        n_features=n_features,
    )

    for start in range(0, n_features, batch_size):
        end = min(n_features, start + batch_size)
        ref_chunk = load_feature_chunk_internal_yxz(ref_lr_feats, start=start, end=end)
        cand_chunk = load_feature_chunk_internal_yxz(cand_lr_feats, start=start, end=end)

        for ref_id, ref_metrics in metrics_by_ref.items():
            if ref_metrics.candidate_ids.size == 0:
                continue
            for candidate_index, _cand_id in enumerate(ref_metrics.candidate_ids.tolist()):
                if ref_metrics.overlap_counts[candidate_index] <= 1:
                    continue

                ref_coords, cand_coords = overlap_cache[ref_id][candidate_index]
                if ref_coords.shape[0] <= 1:
                    continue

                ref_values = sample_feature_chunk_at_internal_coords(ref_chunk, ref_axis_maps, ref_coords)
                cand_values = sample_feature_chunk_at_internal_coords(cand_chunk, cand_axis_maps, cand_coords)

                diff = ref_values - cand_values
                ref_metrics.mse[start:end, candidate_index] = np.mean(
                    diff * diff,
                    axis=1,
                    dtype=np.float32,
                )
                ref_metrics.corr[start:end, candidate_index] = rowwise_correlation(ref_values, cand_values)

    return metrics_by_ref


def build_summary_from_metrics(
    metrics_by_ref: dict[int, RefCandidateMetrics],
    *,
    available_candidates: dict[int, set[int]] | None = None,
    dice_threshold: float,
    corr_threshold: float,
) -> list[RefSummary]:
    summaries: list[RefSummary] = []
    for ref_id in sorted(metrics_by_ref):
        metrics = metrics_by_ref[ref_id]
        if available_candidates is None:
            allowed = np.ones(metrics.candidate_ids.shape[0], dtype=bool)
        else:
            allowed_ids = available_candidates.get(ref_id, set())
            allowed = np.array([int(candidate_id) in allowed_ids for candidate_id in metrics.candidate_ids.tolist()])

        candidates: dict[int, CandidateVote] = {}
        for candidate_id, distance, dice in zip(
            metrics.candidate_ids.tolist(),
            metrics.distances.tolist(),
            metrics.dice.tolist(),
        ):
            if available_candidates is not None and int(candidate_id) not in available_candidates.get(ref_id, set()):
                continue
            candidates[int(candidate_id)] = CandidateVote(
                candidate_label=int(candidate_id),
                wins=0,
                features=[],
                distance=float(distance),
                dice=float(dice),
            )

        if np.any(allowed):
            for feature_index in range(metrics.corr.shape[0]):
                good = (
                    allowed
                    & np.isfinite(metrics.dice)
                    & (metrics.dice > dice_threshold)
                    & np.isfinite(metrics.corr[feature_index])
                    & (metrics.corr[feature_index] > corr_threshold)
                    & np.isfinite(metrics.mse[feature_index])
                )
                if not np.any(good):
                    continue
                good_indices = np.where(good)[0]
                chosen_index = int(good_indices[np.argmin(metrics.mse[feature_index, good_indices])])
                chosen_candidate = int(metrics.candidate_ids[chosen_index])
                candidates[chosen_candidate].wins += 1
                candidates[chosen_candidate].features.append(feature_index)

        ordered = sorted(
            candidates.values(),
            key=lambda candidate: (
                -candidate.wins,
                np.inf if candidate.distance is None else candidate.distance,
                candidate.candidate_label,
            ),
        )
        summaries.append(RefSummary(ref_label=ref_id, candidates=ordered))
    return summaries


def _remove_assigned_from_candidate_pool(
    candidate_pool: dict[int, set[int]],
    *,
    assigned_refs: Iterable[int],
    assigned_cands: Iterable[int],
) -> dict[int, set[int]]:
    assigned_refs_set = {int(value) for value in assigned_refs}
    assigned_cands_set = {int(value) for value in assigned_cands}
    out: dict[int, set[int]] = {}
    for ref_id, candidates in candidate_pool.items():
        if ref_id in assigned_refs_set:
            continue
        out[ref_id] = {int(candidate) for candidate in candidates if int(candidate) not in assigned_cands_set}
    return out


def _finite_mean(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.mean(finite, dtype=np.float64))


def _candidate_metric_index(metrics: RefCandidateMetrics, candidate_id: int) -> int | None:
    matches = np.flatnonzero(metrics.candidate_ids == int(candidate_id))
    if matches.size == 0:
        return None
    return int(matches[0])


def _centroid_sort_tuple(values: np.ndarray) -> tuple[float, float, float]:
    if values.shape[0] < 3:
        return (0.0, 0.0, 0.0)
    return (float(values[0]), float(values[1]), float(values[2]))


def _assignment_sort_key(
    metrics_by_ref: dict[int, RefCandidateMetrics],
    item: tuple[float, int, int],
) -> tuple[float, tuple[float, float, float], tuple[float, float, float], int, int]:
    distance, ref_id, cand_id = item
    ref_centroid = (0.0, 0.0, 0.0)
    cand_centroid = (0.0, 0.0, 0.0)
    metrics = metrics_by_ref.get(int(ref_id))
    if metrics is not None:
        ref_centroid = _centroid_sort_tuple(metrics.ref_centroid)
        candidate_index = _candidate_metric_index(metrics, int(cand_id))
        if candidate_index is not None:
            cand_centroid = _centroid_sort_tuple(metrics.candidate_centroids[candidate_index])
    return (float(distance), ref_centroid, cand_centroid, int(ref_id), int(cand_id))


def _feature_assignment_sort_key(
    metrics_by_ref: dict[int, RefCandidateMetrics],
    item: tuple[int, int, int, float],
) -> tuple[int, float, tuple[float, float, float], tuple[float, float, float], int, int]:
    ref_id, cand_id, wins, distance = item
    distance_key, ref_centroid, cand_centroid, ref_label, cand_label = _assignment_sort_key(
        metrics_by_ref,
        (distance, ref_id, cand_id),
    )
    return (-int(wins), distance_key, ref_centroid, cand_centroid, ref_label, cand_label)


def _candidate_votes_from_summary(summary: list[RefSummary], ref_id: int, candidate_id: int) -> int | None:
    for item in summary:
        if int(item.ref_label) != int(ref_id):
            continue
        for candidate in item.candidates:
            if int(candidate.candidate_label) == int(candidate_id):
                return int(candidate.wins)
        return None
    return None


def _build_assignment_record(
    metrics_by_ref: dict[int, RefCandidateMetrics],
    *,
    ref_id: int,
    cand_id: int,
    method: str,
    stage: int,
    feat_votes: int | None,
    vote_threshold: int | None = None,
) -> AssignmentRecord:
    distance: float | None = None
    dice: float | None = None
    corr: float | None = None
    mse: float | None = None

    metrics = metrics_by_ref.get(int(ref_id))
    if metrics is not None:
        candidate_index = _candidate_metric_index(metrics, int(cand_id))
        if candidate_index is not None:
            distance = float(metrics.distances[candidate_index])
            dice_value = float(metrics.dice[candidate_index])
            dice = dice_value if np.isfinite(dice_value) else None
            corr = _finite_mean(metrics.corr[:, candidate_index])
            mse = _finite_mean(metrics.mse[:, candidate_index])

    return AssignmentRecord(
        ref_label=int(ref_id),
        candidate_label=int(cand_id),
        method=method,
        stage=int(stage),
        distance=distance,
        wins=feat_votes,
        vote_threshold=vote_threshold,
        dice=dice,
        corr=corr,
        mse=mse,
    )


def _assign_aggressive_feature_matches(
    metrics_by_ref: dict[int, RefCandidateMetrics],
    summary_current: list[RefSummary],
    *,
    min_feature_votes: int,
) -> tuple[list[AssignmentRecord], set[int], set[int]]:
    feature_pairs: list[tuple[int, int, int, float]] = []
    for item in summary_current:
        for candidate in item.candidates:
            if int(candidate.wins) < min_feature_votes:
                continue
            distance = np.inf if candidate.distance is None else float(candidate.distance)
            feature_pairs.append((int(item.ref_label), int(candidate.candidate_label), int(candidate.wins), distance))

    assignments: list[AssignmentRecord] = []
    assigned_refs: set[int] = set()
    assigned_cands: set[int] = set()
    for ref_id, cand_id, wins, distance in sorted(
        feature_pairs,
        key=lambda item: _feature_assignment_sort_key(metrics_by_ref, item),
    ):
        if ref_id in assigned_refs or cand_id in assigned_cands:
            continue
        assignment = _build_assignment_record(
            metrics_by_ref,
            ref_id=ref_id,
            cand_id=cand_id,
            method="aggressive_feature_votes",
            stage=2,
            feat_votes=wins,
            vote_threshold=min_feature_votes,
        )
        assignment.distance = distance
        assignments.append(assignment)
        assigned_refs.add(ref_id)
        assigned_cands.add(cand_id)

    return assignments, assigned_refs, assigned_cands


def run_assignment_logic(
    metrics_by_ref: dict[int, RefCandidateMetrics],
    *,
    min_distance_to_remove_cand: float,
    vote_thresholds: tuple[int, ...],
    dice_threshold: float,
    corr_threshold: float,
    ignore_features: bool = False,
    disable_centroid_fallback: bool = False,
    aggressive_feature_matching: bool = False,
    min_feature_votes: int = 1,
) -> tuple[list[AssignmentRecord], list[RefSummary], list[list[RefSummary]], list[dict[str, float]]]:
    all_ref_ids = sorted(metrics_by_ref)
    candidate_pool: dict[int, set[int]] = {
        ref_id: set(metrics.candidate_ids.tolist()) for ref_id, metrics in metrics_by_ref.items()
    }

    initial_summary = build_summary_from_metrics(
        metrics_by_ref,
        available_candidates=candidate_pool,
        dice_threshold=dice_threshold,
        corr_threshold=corr_threshold,
    )
    summary_history: list[list[RefSummary]] = [initial_summary]
    assignments: list[AssignmentRecord] = []
    summary_distance_prefilter: list[dict[str, float]] = []

    nearest_pairs: list[tuple[float, int, int]] = []
    for ref_id in all_ref_ids:
        metrics = metrics_by_ref[ref_id]
        if metrics.candidate_ids.size == 0:
            continue
        best_index = int(np.argmin(metrics.distances))
        min_distance = float(metrics.distances[best_index])
        if min_distance < min_distance_to_remove_cand:
            nearest_pairs.append((min_distance, ref_id, int(metrics.candidate_ids[best_index])))
        else:
            summary_distance_prefilter.append({"RefLabel": float(ref_id), "MinDistance": min_distance})

    assigned_refs: set[int] = set()
    assigned_cands: set[int] = set()
    for distance, ref_id, cand_id in sorted(nearest_pairs, key=lambda item: _assignment_sort_key(metrics_by_ref, item)):
        if ref_id in assigned_refs or cand_id in assigned_cands:
            continue
        feat_votes = None if ignore_features else _candidate_votes_from_summary(initial_summary, ref_id, cand_id)
        assignment = _build_assignment_record(
            metrics_by_ref,
            ref_id=ref_id,
            cand_id=cand_id,
            method="distance_prefilter",
            stage=1,
            feat_votes=feat_votes,
        )
        assignment.distance = distance
        assignments.append(assignment)
        assigned_refs.add(ref_id)
        assigned_cands.add(cand_id)

    candidate_pool = _remove_assigned_from_candidate_pool(
        candidate_pool,
        assigned_refs=assigned_refs,
        assigned_cands=assigned_cands,
    )
    summary_current = build_summary_from_metrics(
        metrics_by_ref,
        available_candidates=candidate_pool,
        dice_threshold=dice_threshold,
        corr_threshold=corr_threshold,
    )
    summary_history.append(summary_current)

    if not ignore_features:
        used_vote_thresholds = tuple(sorted({int(value) for value in vote_thresholds if int(value) > 0}, reverse=True))
        for threshold in used_vote_thresholds:
            while True:
                if not summary_current:
                    break

                top_candidates: list[tuple[int, int, int]] = []
                for item in summary_current:
                    if not item.candidates:
                        continue
                    top = item.candidates[0]
                    if top.wins >= threshold:
                        top_candidates.append((item.ref_label, top.candidate_label, top.wins))

                if not top_candidates:
                    break

                counts = defaultdict(int)
                for _ref_id, cand_id, _wins in top_candidates:
                    counts[cand_id] += 1

                accepted = [
                    (ref_id, cand_id, wins)
                    for ref_id, cand_id, wins in top_candidates
                    if counts[cand_id] == 1
                ]
                if not accepted:
                    break

                new_refs: set[int] = set()
                new_cands: set[int] = set()
                for ref_id, cand_id, wins in accepted:
                    if ref_id in new_refs or cand_id in new_cands:
                        continue
                    assignments.append(
                        _build_assignment_record(
                            metrics_by_ref,
                            ref_id=ref_id,
                            cand_id=cand_id,
                            method=f"vote_threshold_{threshold}",
                            stage=2,
                            feat_votes=wins,
                            vote_threshold=threshold,
                        )
                    )
                    new_refs.add(ref_id)
                    new_cands.add(cand_id)

                if not new_refs:
                    break

                candidate_pool = _remove_assigned_from_candidate_pool(
                    candidate_pool,
                    assigned_refs=new_refs,
                    assigned_cands=new_cands,
                )
                summary_current = build_summary_from_metrics(
                    metrics_by_ref,
                    available_candidates=candidate_pool,
                    dice_threshold=dice_threshold,
                    corr_threshold=corr_threshold,
                )
                summary_history.append(summary_current)

        if aggressive_feature_matching:
            new_assignments, new_refs, new_cands = _assign_aggressive_feature_matches(
                metrics_by_ref,
                summary_current,
                min_feature_votes=int(min_feature_votes),
            )
            if new_assignments:
                assignments.extend(new_assignments)
                candidate_pool = _remove_assigned_from_candidate_pool(
                    candidate_pool,
                    assigned_refs=new_refs,
                    assigned_cands=new_cands,
                )
                summary_current = build_summary_from_metrics(
                    metrics_by_ref,
                    available_candidates=candidate_pool,
                    dice_threshold=dice_threshold,
                    corr_threshold=corr_threshold,
                )
                summary_history.append(summary_current)

    remaining_pairs: list[tuple[float, int, int]] = []
    used_refs = {assignment.ref_label for assignment in assignments}
    used_cands = {assignment.candidate_label for assignment in assignments}
    if not disable_centroid_fallback:
        for ref_id, remaining_candidates in candidate_pool.items():
            if not remaining_candidates:
                continue
            metrics = metrics_by_ref[ref_id]
            for candidate_index, candidate_id in enumerate(metrics.candidate_ids.tolist()):
                if int(candidate_id) in remaining_candidates:
                    remaining_pairs.append((float(metrics.distances[candidate_index]), ref_id, int(candidate_id)))

        for distance, ref_id, cand_id in sorted(
            remaining_pairs,
            key=lambda item: _assignment_sort_key(metrics_by_ref, item),
        ):
            if ref_id in used_refs or cand_id in used_cands:
                continue
            feat_votes = None if ignore_features else _candidate_votes_from_summary(summary_current, ref_id, cand_id)
            assignment = _build_assignment_record(
                metrics_by_ref,
                ref_id=ref_id,
                cand_id=cand_id,
                method="global_closest",
                stage=3,
                feat_votes=feat_votes,
            )
            assignment.distance = distance
            assignments.append(assignment)
            used_refs.add(ref_id)
            used_cands.add(cand_id)

        candidate_pool = _remove_assigned_from_candidate_pool(
            candidate_pool,
            assigned_refs=used_refs,
            assigned_cands=used_cands,
        )
        summary_current = build_summary_from_metrics(
            metrics_by_ref,
            available_candidates=candidate_pool,
            dice_threshold=dice_threshold,
            corr_threshold=corr_threshold,
        )
    summary_history.append(summary_current)

    assignments.sort(key=lambda assignment: (assignment.ref_label, assignment.candidate_label))
    return assignments, initial_summary, summary_history, summary_distance_prefilter


def label_row_index(timepoint: PreparedTimepoint, label_id: int) -> int:
    row_index = int(np.searchsorted(timepoint.label_ids, label_id))
    if row_index >= timepoint.label_ids.size or int(timepoint.label_ids[row_index]) != label_id:
        raise PipelineError(f"Label {label_id} not found in {timepoint.paths.name}.")
    return row_index


def build_tracks(prepared_timepoints: list[PreparedTimepoint], pair_results: list[PairResult]) -> list[Track]:
    if not prepared_timepoints:
        return []

    tracks: list[Track] = []
    active: dict[int, int] = {}

    def build_track_point(
        timepoint_idx: int,
        label_id: int,
        *,
        incoming: SegmentDiagnostics | None = None,
    ) -> TrackPoint:
        prepared = prepared_timepoints[timepoint_idx - 1]
        row_index = label_row_index(prepared, label_id)
        geometry = prepared.geometries[int(label_id)]
        return TrackPoint(
            timepoint=timepoint_idx,
            label_id=int(label_id),
            centroid=tuple(float(value) for value in geometry.centroid.tolist()),
            volume=int(geometry.volume),
            amplitude=float(prepared.amplitudes[row_index]),
            incoming=incoming,
        )

    def start_track(timepoint_idx: int, label_id: int) -> int:
        track_id = len(tracks) + 1
        tracks.append(
            Track(
                track_id=track_id,
                points=[build_track_point(timepoint_idx, label_id)],
                start_time=timepoint_idx,
                length=1,
            )
        )
        return track_id

    def ensure_track_point(
        track_id: int,
        timepoint_idx: int,
        label_id: int,
        *,
        incoming: SegmentDiagnostics | None = None,
    ) -> None:
        track = tracks[track_id - 1]
        for point in track.points:
            if point.timepoint == timepoint_idx and point.label_id == label_id:
                if point.incoming is None and incoming is not None:
                    point.incoming = incoming
                return
        track.points.append(build_track_point(timepoint_idx, label_id, incoming=incoming))

    if prepared_timepoints[0].label_ids.size > 0:
        for label_id in prepared_timepoints[0].label_ids.tolist():
            active[int(label_id)] = start_track(1, int(label_id))

    for pair_index, pair_result in enumerate(pair_results, start=1):
        current_tp = prepared_timepoints[pair_index - 1]
        next_tp = prepared_timepoints[pair_index]

        for label_id in current_tp.label_ids.tolist():
            if int(label_id) not in active:
                active[int(label_id)] = start_track(pair_index, int(label_id))

        next_active: dict[int, int] = {}
        assignments_by_ref = {assignment.ref_label: assignment for assignment in pair_result.assignments}
        for ref_label, cand_label in sorted(pair_result.final_assignment.items()):
            if int(ref_label) not in active:
                active[int(ref_label)] = start_track(pair_index, int(ref_label))
            track_id = active[int(ref_label)]
            ensure_track_point(track_id, pair_index, int(ref_label))
            assignment = assignments_by_ref.get(int(ref_label))
            incoming = None
            if assignment is not None:
                incoming = SegmentDiagnostics(
                    stage=int(assignment.stage),
                    assignment_method=assignment.method,
                    dice=assignment.dice,
                    corr=assignment.corr,
                    mse=assignment.mse,
                    feat_votes=assignment.wins,
                    vote_threshold=assignment.vote_threshold,
                    anisotropic_distance=assignment.distance,
                )
            ensure_track_point(track_id, pair_index + 1, int(cand_label), incoming=incoming)
            next_active[int(cand_label)] = track_id

        active = next_active
        if pair_index == len(pair_results):
            existing_last_frame = {
                point.label_id for track in tracks for point in track.points if point.timepoint == pair_index + 1
            }
            for label_id in next_tp.label_ids.tolist():
                if int(label_id) not in existing_last_frame:
                    start_track(pair_index + 1, int(label_id))

    for track in tracks:
        track.points.sort(key=lambda point: point.timepoint)
        track.start_time = min(point.timepoint for point in track.points)
        track.length = len(track.points)

    tracks.sort(key=lambda track: track.track_id)
    return tracks


def build_matlab_style_tracks(tracks: list[Track]) -> list[dict[str, Any]]:
    matlab_tracks: list[dict[str, Any]] = []
    for track in tracks:
        if not track.points:
            continue
        ordered_points = sorted(track.points, key=lambda point: point.timepoint)
        matlab_tracks.append(
            {
                "track_id": int(track.track_id),
                "start": int(track.start_time),
                "x": np.asarray([point.centroid[1] for point in ordered_points], dtype=np.float64),
                "y": np.asarray([point.centroid[0] for point in ordered_points], dtype=np.float64),
                "z": np.asarray([point.centroid[2] for point in ordered_points], dtype=np.float64),
                "A": np.asarray([point.amplitude for point in ordered_points], dtype=np.float64),
                "label_id": np.asarray([point.label_id for point in ordered_points], dtype=np.int64),
                "volume": np.asarray([point.volume for point in ordered_points], dtype=np.int64),
                "incoming": [point.incoming for point in ordered_points],
            }
        )
    return matlab_tracks


def _diagnostic_value(value: Any) -> Any:
    if value is None:
        return CSV_NAN
    if isinstance(value, float) and not np.isfinite(value):
        return CSV_NAN
    return value


def build_export_rows(
    matlab_tracks: Iterable[dict[str, Any]],
    *,
    z0: int,
    invert_z: bool,
    save_extended_results: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for track in matlab_tracks:
        x = np.asarray(track["x"], dtype=float)
        y = np.asarray(track["y"], dtype=float)
        z = np.asarray(track["z"], dtype=float)
        amplitudes = np.asarray(track["A"], dtype=float)
        label_ids = np.asarray(track.get("label_id", []), dtype=np.int64)
        volumes = np.asarray(track.get("volume", []), dtype=np.int64)
        incoming = list(track.get("incoming", []))
        start = int(track["start"])
        track_id = int(track["track_id"])

        if np.isnan(amplitudes).any():
            continue

        length = min(
            int(amplitudes.size),
            int(x.size),
            int(y.size),
            int(z.size),
            int(label_ids.size),
            int(volumes.size),
            len(incoming),
        )
        if length == 0:
            continue

        for frame_index in range(length):
            row: dict[str, Any] = {
                "track_id": track_id,
                "start": start,
                "t": frame_index + 1,
                "x": float(x[frame_index]),
                "y": float(y[frame_index]),
                "z": float(z0 - z[frame_index]) if invert_z else float(z[frame_index]),
                "A": float(amplitudes[frame_index]),
                "track_length": length,
            }
            if save_extended_results:
                diagnostics = incoming[frame_index]
                row.update(
                    {
                        "stage": CSV_NAN,
                        "assignment_method": CSV_NAN,
                        "dice": CSV_NAN,
                        "corr": CSV_NAN,
                        "mse": CSV_NAN,
                        "feat_votes": CSV_NAN,
                        "vote_threshold": CSV_NAN,
                        "anisotropic_distance": CSV_NAN,
                        "label_id": int(label_ids[frame_index]),
                        "volume": int(volumes[frame_index]),
                    }
                )
                if diagnostics is not None:
                    row.update(
                        {
                            "stage": int(diagnostics.stage),
                            "assignment_method": diagnostics.assignment_method,
                            "dice": _diagnostic_value(diagnostics.dice),
                            "corr": _diagnostic_value(diagnostics.corr),
                            "mse": _diagnostic_value(diagnostics.mse),
                            "feat_votes": _diagnostic_value(diagnostics.feat_votes),
                            "vote_threshold": _diagnostic_value(diagnostics.vote_threshold),
                            "anisotropic_distance": _diagnostic_value(diagnostics.anisotropic_distance),
                        }
                    )
            rows.append(row)
    return rows


def save_tracks_csv(rows: Iterable[dict[str, Any]], *, output_path: Path, save_extended_results: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["track_id", "start", "t", "x", "y", "z", "A", "track_length"]
    if save_extended_results:
        fieldnames.extend(
            [
                "stage",
                "assignment_method",
                "dice",
                "corr",
                "mse",
                "feat_votes",
                "vote_threshold",
                "anisotropic_distance",
                "label_id",
                "volume",
            ]
        )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def default_vote_thresholds(n_features: int) -> tuple[int, ...]:
    thresholds = tuple(threshold for threshold in (400, 320, 300, 280, 260) if threshold <= n_features)
    if thresholds:
        return thresholds
    return (max(1, int(round(0.8 * n_features))),)


def validate_timepoint_shapes(prepared_timepoints: list[PreparedTimepoint], *, require_same_raw_z_size: bool) -> int:
    if not prepared_timepoints:
        raise PipelineError("No prepared timepoints available.")

    expected_shape = prepared_timepoints[0].shape_yxz
    expected_z0 = prepared_timepoints[0].raw_z_size
    for prepared in prepared_timepoints[1:]:
        if prepared.shape_yxz != expected_shape:
            raise PipelineError(
                (
                    "Tracking requires all timepoints to share the same volume shape. "
                    f"Got {expected_shape} and {prepared.shape_yxz}."
                )
            )
        if require_same_raw_z_size and prepared.raw_z_size != expected_z0:
            raise PipelineError(
                (
                    "Tracking requires all timepoints to share the same raw Z size for the final z inversion. "
                    f"Got {expected_z0} and {prepared.raw_z_size}."
                )
            )
    return expected_z0


def run_tracking(
    input_path: Path,
    *,
    segmentation_path: Path,
    output_path: Path | None = None,
    output_filename: str | None = None,
    params: TrackingParams,
) -> Path:
    input_path = input_path.expanduser().resolve()
    segmentation_path = segmentation_path.expanduser().resolve()
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input folder does not exist or is not a directory: {input_path}")
    if not segmentation_path.is_dir():
        raise FileNotFoundError(f"Segmentation folder does not exist or is not a directory: {segmentation_path}")
    output_dir = input_path if output_path is None else output_path.expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise FileNotFoundError(f"Output folder exists but is not a directory: {output_dir}")
    output_filename = normalize_output_filename(output_filename)

    discovered = discover_experiment(input_path, segmentation_path=segmentation_path)
    if len(discovered) < 2:
        raise PipelineError("Tracking requires at least 2 timepoints.")

    prepared_timepoints: list[PreparedTimepoint] = []
    print(f"[tracking] Found {len(discovered)} timepoints", flush=True)
    for index, timepoint_paths in enumerate(discovered, start=1):
        print(f"[tracking] Processing {timepoint_paths.name} ({index}/{len(discovered)})", flush=True)
        prepared_timepoints.append(prepare_timepoint(index, timepoint_paths))
        print(f"[tracking] Completed {timepoint_paths.name}", flush=True)

    z0 = validate_timepoint_shapes(prepared_timepoints, require_same_raw_z_size=bool(params.invert_z))
    spatial_radius = (float(params.max_distance_xy), float(params.max_distance_xy), float(params.max_distance_z))
    vote_thresholds = params.vote_thresholds if params.vote_thresholds is not None else default_vote_thresholds(
        DEFAULT_N_PATCH_FEATURES
    )

    pair_results: list[PairResult] = []
    for pair_index in range(len(prepared_timepoints) - 1):
        ref_tp = prepared_timepoints[pair_index]
        cand_tp = prepared_timepoints[pair_index + 1]
        print(
            f"[tracking] Matching {ref_tp.paths.name} -> {cand_tp.paths.name} ({pair_index + 1}/{len(prepared_timepoints) - 1})",
            flush=True,
        )

        metrics_by_ref = compute_pair_metrics(
            ref_tp,
            cand_tp,
            n_features=DEFAULT_N_PATCH_FEATURES,
            spatial_radius=spatial_radius,
            zratio=float(params.z_distance_weight),
            ignore_features=bool(params.ignore_features),
        )
        assignments, initial_summary, summary_history, summary_distance_prefilter = run_assignment_logic(
            metrics_by_ref,
            min_distance_to_remove_cand=float(params.min_distance_to_remove_cand),
            vote_thresholds=vote_thresholds,
            dice_threshold=float(params.dice_threshold),
            corr_threshold=float(params.corr_threshold),
            ignore_features=bool(params.ignore_features),
            disable_centroid_fallback=bool(params.disable_centroid_fallback),
            aggressive_feature_matching=bool(params.aggressive_feature_matching),
            min_feature_votes=int(params.min_feature_votes),
        )
        pair_results.append(
            PairResult(
                t_ref=ref_tp.index,
                t_cand=cand_tp.index,
                initial_summary=initial_summary,
                summary_current=summary_history[-1] if summary_history else [],
                summary_history=summary_history,
                assignments=assignments,
                final_assignment={assignment.ref_label: assignment.candidate_label for assignment in assignments},
                summary_distance_prefilter=summary_distance_prefilter,
            )
        )
        print(
            f"[tracking] Matched {ref_tp.paths.name} -> {cand_tp.paths.name} ({pair_index + 1}/{len(prepared_timepoints) - 1})",
            flush=True,
        )

    tracks = build_tracks(prepared_timepoints, pair_results)
    matlab_tracks = build_matlab_style_tracks(tracks)
    output_rows = build_export_rows(
        matlab_tracks,
        z0=z0,
        invert_z=bool(params.invert_z),
        save_extended_results=bool(params.save_extended_results),
    )
    output_csv_path = output_dir / output_filename
    save_tracks_csv(
        output_rows,
        output_path=output_csv_path,
        save_extended_results=bool(params.save_extended_results),
    )
    print(f"[tracking] Saved tracks to {output_csv_path}", flush=True)
    print("[tracking] Done", flush=True)
    return output_csv_path


def main() -> None:
    args = parse_args()
    if args.max_distance_xy <= 0:
        raise ValueError("Max XY distance must be greater than 0.")
    if args.max_distance_z <= 0:
        raise ValueError("Max Z distance must be greater than 0.")
    if args.z_distance_weight <= 0:
        raise ValueError("Z distance weight must be greater than 0.")
    if args.min_distance_to_remove_cand < 0:
        raise ValueError("Immediate assignment distance must be nonnegative.")
    if args.dice_threshold < 0 or args.dice_threshold > 1:
        raise ValueError("Dice threshold must be between 0 and 1.")
    if args.corr_threshold < -1 or args.corr_threshold > 1:
        raise ValueError("Correlation threshold must be between -1 and 1.")
    if args.min_feature_votes <= 0:
        raise ValueError("Minimum feature votes must be greater than 0.")
    vote_thresholds = parse_thresholds(args.vote_thresholds)
    if vote_thresholds is not None and any(value <= 0 for value in vote_thresholds):
        raise ValueError("Vote thresholds must be positive integers.")

    input_path = Path(args.input_path).expanduser().resolve()
    segmentation_path = Path(args.segmentation_path).expanduser().resolve()
    run_tracking(
        input_path,
        segmentation_path=segmentation_path,
        output_path=Path(args.output_path).expanduser().resolve() if args.output_path else None,
        output_filename=args.output_filename,
        params=TrackingParams(
            max_distance_xy=float(args.max_distance_xy),
            max_distance_z=float(args.max_distance_z),
            z_distance_weight=float(args.z_distance_weight),
            min_distance_to_remove_cand=float(args.min_distance_to_remove_cand),
            vote_thresholds=vote_thresholds,
            dice_threshold=float(args.dice_threshold),
            corr_threshold=float(args.corr_threshold),
            invert_z=bool(args.invert_z),
            save_extended_results=bool(args.save_extended_results),
            ignore_features=bool(args.ignore_features),
            disable_centroid_fallback=bool(args.disable_centroid_fallback),
            aggressive_feature_matching=bool(args.aggressive_feature_matching),
            min_feature_votes=int(args.min_feature_votes),
        ),
    )


if __name__ == "__main__":
    main()
