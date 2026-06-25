import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";

import ParameterHelpLabel from "../components/ParameterHelpLabel";
import { useJobs } from "../components/JobsProvider";
import ServerDirectoryPicker from "../components/ServerDirectoryPicker";
import { getClientId } from "../lib/clientId";

export type PostProcessingPageKind = "post_processing" | "segmentation" | "tracking";
type WorkflowOption = "process_features" | "foreground_probability_map" | "segmentation" | "tracking";
type PostProcessingMode = "pca" | "high_resolution_features" | "feature_statistics" | "foreground_probability_map";
type SegmentationMode = "voronoi_otsu" | "general_segmentation" | "legacy_probability_map";
type SaveFormat = ".npy" | ".tif";
type GeneralSegmentationSourceKind = "raw" | "probmap" | "pca" | "feature_stats";
type GeneralSegmentationDisplaySource = "raw" | "evaluation";
type GeneralSegmentationDataOperationType =
  | "invert_lut"
  | "subtract_background"
  | "gaussian_smoothing"
  | "laplacian_of_gaussian"
  | "percentile_clipping"
  | "median_filter"
  | "difference_of_gaussians"
  | "top_hat"
  | "black_hat";
type GeneralSegmentationMaskOperationType =
  | "remove_small_objects"
  | "fill_small_holes"
  | "binary_closing"
  | "binary_opening"
  | "dilate"
  | "erode"
  | "remove_border_objects"
  | "size_range";
type GeneralSegmentationInstanceMethod =
  | "none"
  | "connected_components"
  | "voronoi_otsu"
  | "distance_transform_watershed"
  | "intensity_prominence_watershed";
type GeneralSegmentationLogResponse = "bright" | "dark";

type GeneralSegmentationDataOperation = {
  id: string;
  type: GeneralSegmentationDataOperationType;
  radius: string;
  radiusZ: string;
  radiusY: string;
  radiusX: string;
  sigma: string;
  sigmaZ: string;
  sigmaY: string;
  sigmaX: string;
  sigma2: string;
  sigma2Z: string;
  sigma2Y: string;
  sigma2X: string;
  response: GeneralSegmentationLogResponse;
  lowPercentile: string;
  highPercentile: string;
  rescale: boolean;
  outputMin: string;
  outputMax: string;
  anisotropic: boolean;
};

type GeneralSegmentationMaskOperation = {
  id: string;
  type: GeneralSegmentationMaskOperationType;
  size: string;
  minSize: string;
  maxSize: string;
  radius: string;
};

type SerializedGeneralSegmentationDataOperation =
  | { type: "invert_lut" }
  | { type: "subtract_background"; radius: number }
  | { type: "subtract_background"; radius_z: number; radius_y: number; radius_x: number }
  | { type: "gaussian_smoothing"; sigma: number }
  | { type: "gaussian_smoothing"; sigma_z: number; sigma_y: number; sigma_x: number }
  | { type: "laplacian_of_gaussian"; sigma: number; response: GeneralSegmentationLogResponse }
  | {
      type: "laplacian_of_gaussian";
      sigma_z: number;
      sigma_y: number;
      sigma_x: number;
      response: GeneralSegmentationLogResponse;
    }
  | {
      type: "percentile_clipping";
      low_percentile: number;
      high_percentile: number;
      rescale: boolean;
      output_min: number;
      output_max: number;
    }
  | { type: "median_filter"; radius: number }
  | { type: "median_filter"; radius_z: number; radius_y: number; radius_x: number }
  | { type: "difference_of_gaussians"; sigma: number; sigma2: number; response: GeneralSegmentationLogResponse }
  | {
      type: "difference_of_gaussians";
      sigma_z: number;
      sigma_y: number;
      sigma_x: number;
      sigma2_z: number;
      sigma2_y: number;
      sigma2_x: number;
      response: GeneralSegmentationLogResponse;
    }
  | { type: "top_hat"; radius: number }
  | { type: "top_hat"; radius_z: number; radius_y: number; radius_x: number }
  | { type: "black_hat"; radius: number }
  | { type: "black_hat"; radius_z: number; radius_y: number; radius_x: number };

type SerializedGeneralSegmentationMaskOperation =
  | { type: "remove_small_objects"; size: number }
  | { type: "fill_small_holes"; size: number }
  | { type: "binary_closing"; radius: number }
  | { type: "binary_opening"; radius: number }
  | { type: "dilate"; radius: number }
  | { type: "erode"; radius: number }
  | { type: "remove_border_objects" }
  | { type: "size_range"; min_size: number; max_size: number };

type GpuOption = {
  index: number;
  name: string;
};

type PostProcessingOptionsResponse = {
  gpus: GpuOption[];
  gpuError?: string | null;
};

type PostProcessingValidationSuccess = {
  valid: true;
  message: string;
  subfolderCount: number;
  subfolderNames?: string[];
  probmapDensitiesPath?: string;
  probmapDensitiesExists?: boolean;
  availableSources?: GeneralSegmentationSource[];
  sourceWarnings?: GeneralSegmentationSourceWarning[];
};

type PostProcessingValidationFailure = {
  valid: false;
  reasonCode: string;
  message: string;
};

type PostProcessingValidationResult = PostProcessingValidationSuccess | PostProcessingValidationFailure;

type ProcessFeaturesRunRequest = {
  input_path: string;
  output_path: string;
  gpu_index: number | null;
  file_range: {
    start: number | null;
    end: number | null;
  };
  save_high_resolution_features: boolean;
  high_resolution_save_format: SaveFormat;
  save_pca: boolean;
  pca_components: number;
  pca_save_format: SaveFormat;
  global_pca: boolean;
  save_feature_statistics: boolean;
  ignore_trailing_channels: boolean;
  trailing_channels: number;
};

type SegmentationRunRequest = {
  input_path: string;
  output_path: string | null;
  densities_path: string | null;
  gpu_index: number | null;
  mode: SegmentationMode;
  source_id: string;
  threshold: number | null;
  component_index: number;
  invert_mask: boolean;
  data_operations: SerializedGeneralSegmentationDataOperation[];
  mask_operations: SerializedGeneralSegmentationMaskOperation[];
  instance_method: GeneralSegmentationInstanceMethod;
  voronoi_spot_sigma: number;
  voronoi_outline_sigma: number;
  distance_transform_dynamic: number;
  distance_transform_connectivity: number;
  distance_transform_spacing_z: number;
  distance_transform_spacing_y: number;
  distance_transform_spacing_x: number;
  intensity_prominence: number;
  intensity_smoothing_sigma: number;
  intensity_low_percentile: number;
  intensity_high_percentile: number;
  intensity_connectivity: number;
  gaussian_blur_sigma: number;
  rolling_ball_radius: number;
  run_density_estimation: boolean;
  run_stage_2: boolean;
  training_timepoint: string | null;
  seg_tif: string | null;
  valid_mask_tif: string | null;
  density_method: "gpu-hist" | "kde";
  feature_batch: number;
  kde_points: number;
  kde_max_samples: number;
  kde_bandwidth: number | null;
  hist_sigma_bins: number;
  bg_prob_threshold: number;
  fg_prob_threshold: number;
  probmap_threshold: number;
  run_connected_components: boolean;
  seed: number;
};

type ForegroundProbabilityMapRunRequest = {
  input_path: string;
  output_path: string | null;
  densities_path: string | null;
  gpu_index: number | null;
  run_density_estimation: boolean;
  training_timepoint: string | null;
  seg_tif: string | null;
  valid_mask_tif: string | null;
  density_method: "gpu-hist" | "kde";
  feature_batch: number;
  kde_points: number;
  kde_max_samples: number;
  kde_bandwidth: number | null;
  hist_sigma_bins: number;
  seed: number;
};

type TrackingRunRequest = {
  input_path: string;
  segmentation_path: string;
  output_path: string;
  output_filename: string;
  max_distance_xy: number;
  max_distance_z: number;
  z_distance_weight: number;
  min_distance_to_remove_cand: number;
  vote_thresholds: string;
  dice_threshold: number;
  corr_threshold: number;
  save_extended_results: boolean;
  ignore_features: boolean;
  disable_centroid_fallback: boolean;
  aggressive_feature_matching: boolean;
  min_feature_votes: number;
};

type RunSubmittedResponse = {
  submitted: true;
  jobId: string;
  message: string;
};

type RunInvalidResponse = {
  submitted: false;
  valid: false;
  reasonCode: string;
  message: string;
};

type RunResponse = RunSubmittedResponse | RunInvalidResponse;

type RunFeedback = {
  tone: "success" | "error";
  message: string;
};

type GeneralSegmentationPreviewView = "slice" | "max_projection" | "min_projection";
type GeneralSegmentationDisplayContrast = "auto" | "full_range";

type GeneralSegmentationPreviewShape = {
  z: number;
  y: number;
  x: number;
};

type GeneralSegmentationSource = {
  id: string;
  kind: GeneralSegmentationSourceKind;
  label: string;
  folderName: string;
  componentCount: number;
  componentNames?: string[];
  thresholdMin?: number | null;
  thresholdMax?: number | null;
  thresholdStep?: number | null;
};

type GeneralSegmentationSourceWarning = {
  folderName: string;
  message: string;
};

type GeneralSegmentationPreviewTimepoint = {
  name: string;
  compatible: boolean;
  message?: string;
  rawShape?: GeneralSegmentationPreviewShape;
  shape?: GeneralSegmentationPreviewShape;
  zCount?: number;
  width?: number;
  height?: number;
};

type GeneralSegmentationPreviewMetadataResponse = {
  valid: boolean;
  message: string;
  reasonCode?: string;
  subfolderCount?: number;
  subfolderNames?: string[];
  availableSources?: GeneralSegmentationSource[];
  sourceWarnings?: GeneralSegmentationSourceWarning[];
  timepoints?: GeneralSegmentationPreviewTimepoint[];
  defaultTimepoint?: string | null;
};

type GeneralSegmentationPreviewSurfaceResponse = {
  timepoint: string;
  sourceId: string;
  view: GeneralSegmentationPreviewView;
  zIndex: number | null;
  width: number;
  height: number;
  shape: GeneralSegmentationPreviewShape;
  display: {
    dtype: "uint8";
    data: string;
    displayLow: number;
    displayHigh: number;
  };
  evaluation:
    | {
        kind: "slice";
        values: {
          dtype: "float32";
          data: string;
        };
      }
    | {
        kind: "projection";
        maxValues: {
          dtype: "float32";
          data: string;
        };
        minValues: {
          dtype: "float32";
          data: string;
        };
      };
};

type GeneralSegmentationPreviewDataResponse = {
  timepoint: string;
  sourceId: string;
  shape: GeneralSegmentationPreviewShape;
  rangeMin: number;
  rangeMax: number;
  step: number;
};

type GeneralSegmentationPreviewMaskResponse = {
  timepoint: string;
  sourceId: string;
  view: GeneralSegmentationPreviewView;
  zIndex: number | null;
  width: number;
  height: number;
  shape: GeneralSegmentationPreviewShape;
  mask: {
    dtype: "uint32";
    data: string;
    instance: boolean;
  };
};

type GeneralSegmentationPreviewFrame = {
  key: string;
  evaluationKey: string;
  timepoint: string;
  sourceId: string;
  view: GeneralSegmentationPreviewView;
  zIndex: number | null;
  width: number;
  height: number;
  shape: GeneralSegmentationPreviewShape;
  display: Uint8Array;
  evaluationKind: "slice" | "projection";
  evaluationValues: Float32Array | null;
  evaluationMaxValues: Float32Array | null;
  evaluationMinValues: Float32Array | null;
  displayLow: number;
  displayHigh: number;
};

type GeneralSegmentationThresholdRange = {
  key: string;
  min: number;
  max: number;
  step: number;
};

type GeneralSegmentationPreviewMaskFrame = {
  key: string;
  evaluationKey: string;
  timepoint: string;
  sourceId: string;
  view: GeneralSegmentationPreviewView;
  zIndex: number | null;
  width: number;
  height: number;
  shape: GeneralSegmentationPreviewShape;
  mask: Uint32Array;
  maskIsInstance: boolean;
};

const POST_PROCESSING_OPTIONS: Array<{
  value: PostProcessingMode;
  label: string;
}> = [
  {
    value: "pca",
    label: "PCA",
  },
  {
    value: "high_resolution_features",
    label: "High-resolution features",
  },
  {
    value: "feature_statistics",
    label: "Feature statistics",
  },
  {
    value: "foreground_probability_map",
    label: "Foreground probability map",
  },
];

const SEGMENTATION_OPTIONS: Array<{
  value: SegmentationMode;
  label: string;
}> = [
  {
    value: "voronoi_otsu",
    label: "Legacy Voronoi-Otsu",
  },
  {
    value: "general_segmentation",
    label: "General segmentation",
  },
  {
    value: "legacy_probability_map",
    label: "Legacy probability map",
  },
];

const INFERENCE_OUTPUT_FOLDER_DESCRIPTION =
  "folder containing the outputs of a SpatialDINO run, including lr_feats/ and raw/";

const POST_PROCESSING_PARAMETER_HELP = {
  inferenceOutputFolder: INFERENCE_OUTPUT_FOLDER_DESCRIPTION,
  processFeaturesOutputFolder:
    "Root folder where Process features saves pca_<n>/, hr_feats/, and feature_stats/ outputs.",
  chosenFiles: "Limit processing to a contiguous range of validated timepoints. End file is inclusive.",
  segmentationOutputFolder:
    "Root folder where segmentation saves seg_voronoi/, seg_general/, seg_probmap_legacy/, and probmap_densities.npz.",
  foregroundProbabilityMapOutputFolder:
    "Root folder where foreground probability-map generation saves probmap/ and probmap_densities.npz.",
  trackingOutputFolder: "Folder where tracking saves the output CSV.",
  trackingOutputFilename: "CSV file name to write inside the selected output folder.",
  segmentationFolder: "Folder containing one segmentation mask per timepoint, named <timepoint>.tif.",
  selectGpu: "Choose the GPU that will run this post-processing step.",
  saveHighResolutionFeatures: "Write one full-resolution feature volume per channel under hr_feats/<timepoint>/.",
  featureStatistics:
    "Write compact high-resolution feature-channel statistics under feature_stats/ without saving every feature volume.",
  ignoreTrailingChannels: "Exclude trailing attention-style channels before computing feature statistics.",
  trailingChannels: "Number of final feature channels to exclude when enabled.",
  saveFormat: "Choose whether the generated outputs are saved as NumPy arrays or TIFF volumes.",
  savePca: "Export PCA-compressed feature volumes under pca_<n_components>/.",
  globalPca: "Use one PCA basis and one intensity scale across the chosen timepoints.",
  components: "Set how many principal components to keep in the PCA export.",
  voronoiOtsu: "Tune the classical Voronoi-Otsu segmentation pipeline.",
  gaussianBlurSigma: "Smooth the image before seed detection.",
  rollingBallRadius: "Set the background-removal scale used before thresholding.",
  densityEstimationToggle: "Run density estimation now instead of reusing an existing density file.",
  stage1OutputFile: "Path to an existing probmap_densities.npz file produced by Stage 1.",
  trainingTimepoint: "Select the timepoint used to fit the probability-map density model.",
  segmentationTif:
    "Annotated binary foreground/background mask used in Stage 1 to fit FG/BG probability distributions over spatialDINO features.",
  validMaskTif: "Optional mask that limits Stage 1 density fitting to valid voxels only.",
  densityEstimationSettings: "Configure how class densities are estimated for probability-map segmentation.",
  method: "Choose the density estimator used to model feature distributions.",
  densityGridSize: "Set the resolution of the density grid used by the estimator.",
  histogramSigma: "Smooth the GPU-histogram density estimate in grid units.",
  kdeBandwidth: "Override KDE smoothing width; leave blank to choose it automatically.",
  sampling: "Control how many feature samples are drawn for density fitting.",
  maxSamplesPerClass: "Cap the number of samples taken from each class.",
  randomSeed: "Make random sampling reproducible.",
  probabilityMapEstimation: "Tune the pass that turns densities into probability maps.",
  featureBatch: "Number of feature chunks processed at once during estimation.",
  bgProbabilityThreshold: "Minimum background probability used to mark voxels as background.",
  fgProbabilityThreshold: "Minimum foreground probability used to accept a cell candidate.",
  generalSegmentationSource: "Choose which validated data source is thresholded to create the mask.",
  generalEvaluationData: "Choose which validated data source is processed and thresholded.",
  generalDisplaySource: "Choose whether the preview shows raw data or the processed evaluation data.",
  generalThresholdComponent: "Choose the source component or statistic used for thresholding.",
  dataProcessing: "Apply operations to the evaluation data before thresholding.",
  maskProcessing: "Apply operations to the binary mask after thresholding.",
  instanceSegmentation: "Choose whether the final mask stays binary or becomes an instance-label image.",
  simpleProbabilityThreshold: "Minimum source value used to mark voxels as foreground.",
  invertMask: "Use values below the threshold as foreground.",
  runConnectedComponents: "Convert the binary foreground mask into connected-component instance labels.",
  matchWindow: "Set the maximum per-frame search distance for track matching.",
  maxXyDistance: "Largest allowed XY displacement between linked detections.",
  maxZDistance: "Largest allowed Z displacement between linked detections.",
  distanceLogic: "Weight and shortcut rules used when resolving candidate matches.",
  zDistanceWeight: "Multiplier applied to Z distance when scoring matches.",
  immediateAssignmentDistance: "Distance below which a candidate is assigned immediately.",
  votingThresholds: "Tune the overlap-and-voting acceptance rules for track linking.",
  voteThresholds: "Comma-separated minimum feature-vote counts checked in order.",
  minDice: "Minimum Dice overlap required for a match.",
  minCorr: "Minimum intensity correlation required for a match.",
  saveExtendedResults: "Append assignment-stage diagnostics and object labels to the tracking CSV.",
  ignoreFeatures: "Skip SpatialDINO feature voting and resolve remaining links by distance only.",
  disableCentroidFallback: "Skip centroid-only fallback; unresolved detections start or end tracks.",
  aggressiveFeatureMatching: "Greedily match remaining candidates using feature votes before centroid fallback.",
  minFeatureVotes: "Minimum feature votes required for aggressive feature matching.",
} as const;

function parentPathForPicker(path: string | null): string | null {
  if (!path) return null;
  const normalized = path.trim();
  if (!normalized.startsWith("/")) return null;
  const slashIndex = normalized.lastIndexOf("/");
  if (slashIndex <= 0) return "/";
  return normalized.slice(0, slashIndex);
}

function createGeneralDataOperation(
  type: GeneralSegmentationDataOperationType = "gaussian_smoothing"
): GeneralSegmentationDataOperation {
  return {
    id: `data-${Date.now()}-${Math.random().toString(36).slice(2)}`,
    type,
    radius: "10",
    radiusZ: "10",
    radiusY: "10",
    radiusX: "10",
    sigma: "1",
    sigmaZ: "1",
    sigmaY: "1",
    sigmaX: "1",
    sigma2: "2",
    sigma2Z: "2",
    sigma2Y: "2",
    sigma2X: "2",
    response: "bright",
    lowPercentile: "1",
    highPercentile: "99",
    rescale: false,
    outputMin: "0",
    outputMax: "1",
    anisotropic: false,
  };
}

function createGeneralMaskOperation(
  type: GeneralSegmentationMaskOperationType = "remove_small_objects"
): GeneralSegmentationMaskOperation {
  return {
    id: `mask-${Date.now()}-${Math.random().toString(36).slice(2)}`,
    type,
    size: "64",
    minSize: "0",
    maxSize: "0",
    radius: "1",
  };
}

function parseFiniteNumber(value: string): number | null {
  const parsed = Number.parseFloat(value.trim());
  return Number.isFinite(parsed) ? parsed : null;
}

function parseNonnegativeNumber(value: string): number | null {
  const parsed = Number.parseFloat(value.trim());
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : null;
}

function parsePositiveNumber(value: string): number | null {
  const parsed = Number.parseFloat(value.trim());
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
}

function parseNonnegativeInteger(value: string): number | null {
  const parsed = Number.parseInt(value.trim(), 10);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : null;
}

function dataScaleValue(operation: GeneralSegmentationDataOperation, key: "radius" | "sigma" | "sigma2"): string {
  if (key === "radius") return operation.radius;
  if (key === "sigma") return operation.sigma;
  return operation.sigma2;
}

function dataScaleAxisValue(
  operation: GeneralSegmentationDataOperation,
  key: "radius" | "sigma" | "sigma2",
  axis: "Z" | "Y" | "X"
): string {
  if (key === "radius") {
    if (axis === "Z") return operation.radiusZ;
    if (axis === "Y") return operation.radiusY;
    return operation.radiusX;
  }
  if (key === "sigma") {
    if (axis === "Z") return operation.sigmaZ;
    if (axis === "Y") return operation.sigmaY;
    return operation.sigmaX;
  }
  if (axis === "Z") return operation.sigma2Z;
  if (axis === "Y") return operation.sigma2Y;
  return operation.sigma2X;
}

function serializeDataScaleFields(
  operation: GeneralSegmentationDataOperation,
  key: "radius" | "sigma" | "sigma2",
  strict: boolean,
  label: string,
  fallback: number
): { fields: Record<string, number>; error: string | null } {
  if (!operation.anisotropic) {
    const value = parseNonnegativeNumber(dataScaleValue(operation, key));
    if (value === null) {
      if (strict) return { fields: {}, error: `Enter a valid nonnegative ${label}.` };
      return { fields: { [key]: fallback }, error: null };
    }
    return { fields: { [key]: value }, error: null };
  }

  const zValue = parseNonnegativeNumber(dataScaleAxisValue(operation, key, "Z"));
  const yValue = parseNonnegativeNumber(dataScaleAxisValue(operation, key, "Y"));
  const xValue = parseNonnegativeNumber(dataScaleAxisValue(operation, key, "X"));
  if (zValue === null || yValue === null || xValue === null) {
    if (strict) return { fields: {}, error: `Enter valid nonnegative ${label} Z/Y/X values.` };
    return {
      fields: {
        [`${key}_z`]: fallback,
        [`${key}_y`]: fallback,
        [`${key}_x`]: fallback,
      },
      error: null,
    };
  }
  return {
    fields: {
      [`${key}_z`]: zValue,
      [`${key}_y`]: yValue,
      [`${key}_x`]: xValue,
    },
    error: null,
  };
}

function serializeGeneralDataOperations(
  enabled: boolean,
  operations: GeneralSegmentationDataOperation[],
  strict: boolean
): { operations: SerializedGeneralSegmentationDataOperation[]; error: string | null } {
  if (!enabled) return { operations: [], error: null };
  const serialized: SerializedGeneralSegmentationDataOperation[] = [];
  for (const operation of operations) {
    if (operation.type === "invert_lut") {
      serialized.push({ type: "invert_lut" });
    } else if (operation.type === "subtract_background") {
      const scale = serializeDataScaleFields(operation, "radius", strict, "rolling-ball radius", 10);
      if (scale.error) return { operations: [], error: scale.error };
      serialized.push({ type: "subtract_background", ...scale.fields } as SerializedGeneralSegmentationDataOperation);
    } else if (operation.type === "gaussian_smoothing") {
      const scale = serializeDataScaleFields(operation, "sigma", strict, "Gaussian sigma", 1);
      if (scale.error) return { operations: [], error: scale.error };
      serialized.push({ type: "gaussian_smoothing", ...scale.fields } as SerializedGeneralSegmentationDataOperation);
    } else if (operation.type === "laplacian_of_gaussian") {
      const scale = serializeDataScaleFields(operation, "sigma", strict, "LoG sigma", 1);
      if (scale.error) return { operations: [], error: scale.error };
      serialized.push({
        type: "laplacian_of_gaussian",
        ...scale.fields,
        response: operation.response,
      } as SerializedGeneralSegmentationDataOperation);
    } else if (operation.type === "percentile_clipping") {
      const lowPercentile = parseNonnegativeNumber(operation.lowPercentile);
      const highPercentile = parseNonnegativeNumber(operation.highPercentile);
      if (
        lowPercentile === null ||
        highPercentile === null ||
        lowPercentile > 100 ||
        highPercentile > 100 ||
        lowPercentile >= highPercentile
      ) {
        if (strict) return { operations: [], error: "Enter ordered percentile values between 0 and 100." };
        serialized.push({
          type: "percentile_clipping",
          low_percentile: 1,
          high_percentile: 99,
          rescale: operation.rescale,
          output_min: 0,
          output_max: 1,
        });
      } else {
        const outputMin = parseFiniteNumber(operation.outputMin);
        const outputMax = parseFiniteNumber(operation.outputMax);
        if (operation.rescale && (outputMin === null || outputMax === null || outputMin >= outputMax)) {
          if (strict) return { operations: [], error: "Enter ordered finite output range values." };
          serialized.push({
            type: "percentile_clipping",
            low_percentile: lowPercentile,
            high_percentile: highPercentile,
            rescale: true,
            output_min: 0,
            output_max: 1,
          });
        } else {
          serialized.push({
            type: "percentile_clipping",
            low_percentile: lowPercentile,
            high_percentile: highPercentile,
            rescale: operation.rescale,
            output_min: outputMin ?? 0,
            output_max: outputMax ?? 1,
          });
        }
      }
    } else if (operation.type === "median_filter") {
      const scale = serializeDataScaleFields(operation, "radius", strict, "median radius", 1);
      if (scale.error) return { operations: [], error: scale.error };
      serialized.push({ type: "median_filter", ...scale.fields } as SerializedGeneralSegmentationDataOperation);
    } else if (operation.type === "difference_of_gaussians") {
      const sigma = serializeDataScaleFields(operation, "sigma", strict, "DoG small sigma", 1);
      if (sigma.error) return { operations: [], error: sigma.error };
      const sigma2 = serializeDataScaleFields(operation, "sigma2", strict, "DoG large sigma", 2);
      if (sigma2.error) return { operations: [], error: sigma2.error };
      serialized.push({
        type: "difference_of_gaussians",
        ...sigma.fields,
        ...sigma2.fields,
        response: operation.response,
      } as SerializedGeneralSegmentationDataOperation);
    } else if (operation.type === "top_hat" || operation.type === "black_hat") {
      const scale = serializeDataScaleFields(operation, "radius", strict, "hat radius", 10);
      if (scale.error) return { operations: [], error: scale.error };
      serialized.push({ type: operation.type, ...scale.fields } as SerializedGeneralSegmentationDataOperation);
    }
  }
  return { operations: serialized, error: null };
}

function serializeGeneralMaskOperations(
  enabled: boolean,
  operations: GeneralSegmentationMaskOperation[],
  strict: boolean
): { operations: SerializedGeneralSegmentationMaskOperation[]; error: string | null } {
  if (!enabled) return { operations: [], error: null };
  const serialized: SerializedGeneralSegmentationMaskOperation[] = [];
  for (const operation of operations) {
    if (operation.type === "remove_small_objects" || operation.type === "fill_small_holes") {
      const size = parseNonnegativeInteger(operation.size);
      if (size === null) {
        if (strict) return { operations: [], error: "Enter a valid nonnegative mask operation size." };
        serialized.push({ type: operation.type, size: 64 });
      } else {
        serialized.push({ type: operation.type, size });
      }
    } else if (
      operation.type === "binary_closing" ||
      operation.type === "binary_opening" ||
      operation.type === "dilate" ||
      operation.type === "erode"
    ) {
      const radius = parseNonnegativeNumber(operation.radius);
      if (radius === null) {
        if (strict) return { operations: [], error: "Enter a valid nonnegative mask operation radius." };
        serialized.push({ type: operation.type, radius: 1 });
      } else {
        serialized.push({ type: operation.type, radius });
      }
    } else if (operation.type === "remove_border_objects") {
      serialized.push({ type: "remove_border_objects" });
    } else if (operation.type === "size_range") {
      const minSize = parseNonnegativeInteger(operation.minSize);
      const maxSize = parseNonnegativeInteger(operation.maxSize);
      if (minSize === null || maxSize === null || (maxSize > 0 && minSize > maxSize)) {
        if (strict) return { operations: [], error: "Enter a valid mask size range." };
        serialized.push({ type: "size_range", min_size: 0, max_size: 0 });
      } else {
        serialized.push({ type: "size_range", min_size: minSize, max_size: maxSize });
      }
    }
  }
  return { operations: serialized, error: null };
}

type PostProcessingPageProps = {
  pageKind?: PostProcessingPageKind;
};

export default function PostProcessingPage({ pageKind = "post_processing" }: PostProcessingPageProps) {
  const jobs = useJobs();
  const validationRequestIdRef = useRef(0);
  const trackingSegmentationValidationRequestIdRef = useRef(0);

  const [selectedPostProcessingMode, setSelectedPostProcessingMode] = useState<PostProcessingMode>("pca");
  const [selectedSegmentationMode, setSelectedSegmentationMode] = useState<SegmentationMode>("voronoi_otsu");
  const [pickerOpen, setPickerOpen] = useState(false);
  const [pickerTarget, setPickerTarget] = useState<
    | "input"
    | "process_output"
    | "segmentation_output"
    | "tracking_segmentation"
    | "tracking_output"
    | "seg_tif"
    | "valid_mask"
    | "probmap_densities"
  >("input");
  const [inputPath, setInputPath] = useState<string | null>(null);
  const [saveProcessFeaturesInDifferentOutputFolder, setSaveProcessFeaturesInDifferentOutputFolder] = useState(false);
  const [processFeaturesOutputPath, setProcessFeaturesOutputPath] = useState<string | null>(null);
  const [saveSegmentationInDifferentOutputFolder, setSaveSegmentationInDifferentOutputFolder] = useState(false);
  const [segmentationOutputPath, setSegmentationOutputPath] = useState<string | null>(null);
  const [trackingSegmentationPath, setTrackingSegmentationPath] = useState<string | null>(null);
  const [saveTrackingInDifferentOutputFolder, setSaveTrackingInDifferentOutputFolder] = useState(false);
  const [trackingOutputPath, setTrackingOutputPath] = useState<string | null>(null);
  const [trackingOutputFilename, setTrackingOutputFilename] = useState("tracks.csv");
  const [validationLoading, setValidationLoading] = useState(false);
  const [validationResult, setValidationResult] = useState<PostProcessingValidationResult | null>(null);
  const [trackingSegmentationValidationLoading, setTrackingSegmentationValidationLoading] = useState(false);
  const [trackingSegmentationValidationResult, setTrackingSegmentationValidationResult] =
    useState<PostProcessingValidationResult | null>(null);
  const [availableGpus, setAvailableGpus] = useState<GpuOption[]>([]);
  const [gpuError, setGpuError] = useState<string | null>(null);
  const [optionsLoading, setOptionsLoading] = useState(true);
  const [optionsError, setOptionsError] = useState<string | null>(null);
  const [selectedGpuIndex, setSelectedGpuIndex] = useState<number | null>(null);
  const [highResolutionSaveFormat, setHighResolutionSaveFormat] = useState<SaveFormat>(".tif");
  const [pcaComponents, setPcaComponents] = useState("3");
  const [pcaSaveFormat, setPcaSaveFormat] = useState<SaveFormat>(".tif");
  const [globalPca, setGlobalPca] = useState(true);
  const [ignoreTrailingChannels, setIgnoreTrailingChannels] = useState(true);
  const [trailingChannels, setTrailingChannels] = useState("6");
  const [processFeaturesFileRange, setProcessFeaturesFileRange] = useState({ start: "0", end: "" });
  const [gaussianBlurSigma, setGaussianBlurSigma] = useState("3");
  const [rollingBallRadius, setRollingBallRadius] = useState("10");
  const [runDensityEstimation, setRunDensityEstimation] = useState(true);
  const [runProbabilityMapStage2, setRunProbabilityMapStage2] = useState(true);
  const [densityEstimationTimepoint, setDensityEstimationTimepoint] = useState<string>("");
  const [segTifPath, setSegTifPath] = useState<string | null>(null);
  const [validMaskTifPath, setValidMaskTifPath] = useState<string | null>(null);
  const [probmapDensitiesFilePath, setProbmapDensitiesFilePath] = useState<string | null>(null);
  const [probabilityMapDensityMethod, setProbabilityMapDensityMethod] = useState<"gpu-hist" | "kde">("gpu-hist");
  const [probabilityMapFeatureBatch, setProbabilityMapFeatureBatch] = useState("32");
  const [probabilityMapKdePoints, setProbabilityMapKdePoints] = useState("512");
  const [probabilityMapKdeMaxSamples, setProbabilityMapKdeMaxSamples] = useState("200000");
  const [probabilityMapKdeBandwidth, setProbabilityMapKdeBandwidth] = useState("");
  const [probabilityMapHistSigmaBins, setProbabilityMapHistSigmaBins] = useState("1.5");
  const [probabilityMapBgThreshold, setProbabilityMapBgThreshold] = useState("0.4");
  const [probabilityMapFgThreshold, setProbabilityMapFgThreshold] = useState("0.95");
  const [generalSegmentationSourceId, setGeneralSegmentationSourceId] = useState("raw");
  const [generalSegmentationThresholdComponent, setGeneralSegmentationThresholdComponent] = useState(0);
  const [generalSegmentationDisplaySource, setGeneralSegmentationDisplaySource] =
    useState<GeneralSegmentationDisplaySource>("raw");
  const [generalSegmentationProcessData, setGeneralSegmentationProcessData] = useState(false);
  const [generalSegmentationDataOperations, setGeneralSegmentationDataOperations] = useState<
    GeneralSegmentationDataOperation[]
  >([]);
  const [generalSegmentationThreshold, setGeneralSegmentationThreshold] = useState("0.5");
  const [generalSegmentationInvertMask, setGeneralSegmentationInvertMask] = useState(false);
  const [generalSegmentationProcessMask, setGeneralSegmentationProcessMask] = useState(false);
  const [generalSegmentationMaskOperations, setGeneralSegmentationMaskOperations] = useState<
    GeneralSegmentationMaskOperation[]
  >([]);
  const [generalSegmentationInstanceMethod, setGeneralSegmentationInstanceMethod] =
    useState<GeneralSegmentationInstanceMethod>("connected_components");
  const [generalSegmentationVoronoiSpotSigma, setGeneralSegmentationVoronoiSpotSigma] = useState("2");
  const [generalSegmentationVoronoiOutlineSigma, setGeneralSegmentationVoronoiOutlineSigma] = useState("2");
  const [generalSegmentationDistanceDynamic, setGeneralSegmentationDistanceDynamic] = useState("1");
  const [generalSegmentationDistanceConnectivity, setGeneralSegmentationDistanceConnectivity] = useState<6 | 26>(6);
  const [generalSegmentationDistanceSpacingZ, setGeneralSegmentationDistanceSpacingZ] = useState("1");
  const [generalSegmentationDistanceSpacingY, setGeneralSegmentationDistanceSpacingY] = useState("1");
  const [generalSegmentationDistanceSpacingX, setGeneralSegmentationDistanceSpacingX] = useState("1");
  const [generalSegmentationIntensityProminence, setGeneralSegmentationIntensityProminence] = useState("0.15");
  const [generalSegmentationIntensitySmoothingSigma, setGeneralSegmentationIntensitySmoothingSigma] = useState("0");
  const [generalSegmentationIntensityLowPercentile, setGeneralSegmentationIntensityLowPercentile] = useState("1");
  const [generalSegmentationIntensityHighPercentile, setGeneralSegmentationIntensityHighPercentile] = useState("99");
  const [generalSegmentationIntensityConnectivity, setGeneralSegmentationIntensityConnectivity] = useState<6 | 26>(6);
  const [probabilityMapSeed, setProbabilityMapSeed] = useState("1337");
  const [trackingMaxDistanceXy, setTrackingMaxDistanceXy] = useState("35");
  const [trackingMaxDistanceZ, setTrackingMaxDistanceZ] = useState("15");
  const [trackingZDistanceWeight, setTrackingZDistanceWeight] = useState("2.5");
  const [trackingMinDistanceToRemoveCand, setTrackingMinDistanceToRemoveCand] = useState("0");
  const [trackingVoteThresholds, setTrackingVoteThresholds] = useState("360,340,320,300");
  const [trackingDiceThreshold, setTrackingDiceThreshold] = useState("0.5");
  const [trackingCorrThreshold, setTrackingCorrThreshold] = useState("0.5");
  const [trackingSaveExtendedResults, setTrackingSaveExtendedResults] = useState(false);
  const [trackingIgnoreFeatures, setTrackingIgnoreFeatures] = useState(false);
  const [trackingDisableCentroidFallback, setTrackingDisableCentroidFallback] = useState(false);
  const [trackingAggressiveFeatureMatching, setTrackingAggressiveFeatureMatching] = useState(false);
  const [trackingMinFeatureVotes, setTrackingMinFeatureVotes] = useState("1");
  const [voronoiOptionalOpen, setVoronoiOptionalOpen] = useState(false);
  const [probabilityMapOptionalOpen, setProbabilityMapOptionalOpen] = useState(false);
  const [trackingOptionalOpen, setTrackingOptionalOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [runFeedback, setRunFeedback] = useState<RunFeedback | null>(null);

  const selectedWorkflow: WorkflowOption =
    pageKind === "tracking"
      ? "tracking"
      : pageKind === "segmentation"
        ? "segmentation"
        : selectedPostProcessingMode === "foreground_probability_map"
          ? "foreground_probability_map"
          : "process_features";
  const pageTitle =
    pageKind === "tracking" ? "Tracking" : pageKind === "segmentation" ? "Segmentation" : "Post-processing";
  const processFeaturesSelected = selectedWorkflow === "process_features";
  const foregroundProbabilityMapSelected = selectedWorkflow === "foreground_probability_map";
  const segmentationSelected = selectedWorkflow === "segmentation";
  const trackingSelected = selectedWorkflow === "tracking";
  const pcaSelected = pageKind === "post_processing" && selectedPostProcessingMode === "pca";
  const highResolutionFeaturesSelected =
    pageKind === "post_processing" && selectedPostProcessingMode === "high_resolution_features";
  const featureStatisticsSelected =
    pageKind === "post_processing" && selectedPostProcessingMode === "feature_statistics";
  const voronoiOtsuSelected = segmentationSelected && selectedSegmentationMode === "voronoi_otsu";
  const generalSegmentationSelected = segmentationSelected && selectedSegmentationMode === "general_segmentation";
  const legacyProbabilityMapSelected = segmentationSelected && selectedSegmentationMode === "legacy_probability_map";
  const densityControlsSelected = foregroundProbabilityMapSelected || legacyProbabilityMapSelected;

  useEffect(() => {
    let cancelled = false;

    async function loadOptions() {
      setOptionsLoading(true);
      setOptionsError(null);
      try {
        const resp = await fetch("/api/inference/options", { headers: { Accept: "application/json" } });
        if (!resp.ok) {
          throw new Error(`Post-processing options failed: ${resp.status} ${resp.statusText}`);
        }

        const json = (await resp.json()) as PostProcessingOptionsResponse;
        if (cancelled) return;

        setAvailableGpus(json.gpus);
        setGpuError(json.gpuError?.trim() || null);
      } catch (error) {
        if (cancelled) return;
        const message = error instanceof Error ? error.message : "Unknown error";
        setOptionsError(message);
      } finally {
        if (!cancelled) {
          setOptionsLoading(false);
        }
      }
    }

    void loadOptions();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    validationRequestIdRef.current += 1;
    setValidationLoading(false);
    setValidationResult(null);
  }, [inputPath, selectedWorkflow]);

  useEffect(() => {
    setProbmapDensitiesFilePath(null);
  }, [inputPath]);

  useEffect(() => {
    trackingSegmentationValidationRequestIdRef.current += 1;
    setTrackingSegmentationValidationLoading(false);
    setTrackingSegmentationValidationResult(null);
  }, [inputPath, trackingSegmentationPath, selectedWorkflow]);

  useEffect(() => {
    setRunFeedback(null);
  }, [
    pageKind,
    selectedPostProcessingMode,
    selectedWorkflow,
    selectedSegmentationMode,
    inputPath,
    saveProcessFeaturesInDifferentOutputFolder,
    processFeaturesOutputPath,
    saveSegmentationInDifferentOutputFolder,
    segmentationOutputPath,
    trackingSegmentationPath,
    saveTrackingInDifferentOutputFolder,
    trackingOutputPath,
    trackingOutputFilename,
    selectedGpuIndex,
    highResolutionSaveFormat,
    pcaComponents,
    pcaSaveFormat,
    globalPca,
    ignoreTrailingChannels,
    trailingChannels,
    processFeaturesFileRange,
    gaussianBlurSigma,
    rollingBallRadius,
    runDensityEstimation,
    densityEstimationTimepoint,
    segTifPath,
    validMaskTifPath,
    probmapDensitiesFilePath,
    probabilityMapDensityMethod,
    probabilityMapFeatureBatch,
    probabilityMapKdePoints,
    probabilityMapKdeMaxSamples,
    probabilityMapKdeBandwidth,
    probabilityMapHistSigmaBins,
    probabilityMapBgThreshold,
    probabilityMapFgThreshold,
    generalSegmentationSourceId,
    generalSegmentationThresholdComponent,
    generalSegmentationDisplaySource,
    generalSegmentationProcessData,
    generalSegmentationDataOperations,
    generalSegmentationThreshold,
    generalSegmentationInvertMask,
    generalSegmentationProcessMask,
    generalSegmentationMaskOperations,
    generalSegmentationInstanceMethod,
    generalSegmentationVoronoiSpotSigma,
    generalSegmentationVoronoiOutlineSigma,
    probabilityMapSeed,
    trackingMaxDistanceXy,
    trackingMaxDistanceZ,
    trackingZDistanceWeight,
    trackingMinDistanceToRemoveCand,
    trackingVoteThresholds,
    trackingDiceThreshold,
    trackingCorrThreshold,
    trackingSaveExtendedResults,
    trackingIgnoreFeatures,
    trackingDisableCentroidFallback,
    trackingAggressiveFeatureMatching,
    trackingMinFeatureVotes,
  ]);

  const inputValidated = validationResult?.valid === true;
  const trackingSegmentationValidated = trackingSegmentationValidationResult?.valid === true;
  const parametersVisible = trackingSelected ? inputValidated && trackingSegmentationValidated : inputValidated;
  const validatedSubfolderNames = inputValidated ? validationResult.subfolderNames ?? [] : [];
  const generalSegmentationSources = inputValidated ? validationResult.availableSources ?? [] : [];
  const generalSegmentationSourceWarnings = inputValidated ? validationResult.sourceWarnings ?? [] : [];
  const selectedGeneralSegmentationSource =
    generalSegmentationSources.find((source) => source.id === generalSegmentationSourceId) ??
    generalSegmentationSources[0] ??
    null;
  const maxProcessFeaturesFileIndex = processFeaturesSelected && inputValidated ? validationResult.subfolderCount - 1 : undefined;
  const showDensityEstimationTimepoint = validatedSubfolderNames.length > 1;
  const probmapDensitiesPath = inputValidated ? validationResult.probmapDensitiesPath ?? null : null;
  const probmapDensitiesExists = inputValidated ? Boolean(validationResult.probmapDensitiesExists) : false;
  const effectiveProcessFeaturesOutputPath = saveProcessFeaturesInDifferentOutputFolder ? processFeaturesOutputPath : inputPath;
  const effectiveSegmentationOutputPath = saveSegmentationInDifferentOutputFolder ? segmentationOutputPath : inputPath;
  const effectiveTrackingOutputPath = saveTrackingInDifferentOutputFolder ? trackingOutputPath : inputPath;
  const inputValidationSuccessMessage = validationResult?.valid
    ? `Validated inference output folder. Found ${validationResult.subfolderCount} timepoint${
        validationResult.subfolderCount === 1 ? "" : "s"
      }.`
    : null;
  const trackingSegmentationValidationSuccessMessage = trackingSegmentationValidationResult?.valid
    ? `Validated segmentation output folder. Found ${trackingSegmentationValidationResult.subfolderCount} mask file${
        trackingSegmentationValidationResult.subfolderCount === 1 ? "" : "s"
      }.`
    : null;
  const pickerTitle =
    pickerTarget === "input"
      ? "Choose the inference output folder"
      : pickerTarget === "process_output" || pickerTarget === "tracking_output"
        ? "Choose the output folder"
        : pickerTarget === "segmentation_output"
          ? foregroundProbabilityMapSelected
            ? "Choose the output folder"
            : "Choose the segmentation output folder"
          : pickerTarget === "tracking_segmentation"
            ? "Choose the segmentation mask folder"
          : pickerTarget === "probmap_densities"
            ? "Choose the Stage 1 output file"
          : pickerTarget === "seg_tif"
            ? "Choose the annotated FG/BG mask"
            : "Choose the valid voxels mask";
  const pickerInitialPath =
    pickerTarget === "input"
      ? inputPath
      : pickerTarget === "process_output"
        ? effectiveProcessFeaturesOutputPath ?? inputPath
        : pickerTarget === "segmentation_output"
          ? effectiveSegmentationOutputPath ?? inputPath
          : pickerTarget === "tracking_segmentation"
            ? trackingSegmentationPath ?? inputPath
            : pickerTarget === "tracking_output"
              ? effectiveTrackingOutputPath ?? trackingSegmentationPath ?? inputPath
              : pickerTarget === "probmap_densities"
                ? parentPathForPicker(probmapDensitiesFilePath) ?? effectiveSegmentationOutputPath ?? inputPath
              : pickerTarget === "seg_tif"
                ? parentPathForPicker(segTifPath) ?? inputPath
                : parentPathForPicker(validMaskTifPath) ?? inputPath;

  useEffect(() => {
    if (!densityControlsSelected) return;
    if (!validatedSubfolderNames.length) return;
    if (densityEstimationTimepoint && validatedSubfolderNames.includes(densityEstimationTimepoint)) return;
    setDensityEstimationTimepoint(validatedSubfolderNames[0]);
  }, [densityControlsSelected, densityEstimationTimepoint, validatedSubfolderNames]);

  useEffect(() => {
    if (!generalSegmentationSelected || !generalSegmentationSources.length) return;
    if (generalSegmentationSources.some((source) => source.id === generalSegmentationSourceId)) return;
    const rawSource = generalSegmentationSources.find((source) => source.id === "raw");
    setGeneralSegmentationSourceId((rawSource ?? generalSegmentationSources[0]).id);
  }, [generalSegmentationSelected, generalSegmentationSourceId, generalSegmentationSources]);

  useEffect(() => {
    if (!selectedGeneralSegmentationSource) return;
    const maxComponent = Math.max(0, selectedGeneralSegmentationSource.componentCount - 1);
    setGeneralSegmentationThresholdComponent((current) => clampInteger(current, 0, maxComponent));
  }, [selectedGeneralSegmentationSource]);

  useEffect(() => {
    if (!densityControlsSelected || runDensityEstimation) return;
    if (legacyProbabilityMapSelected && !runProbabilityMapStage2) return;
    if (probmapDensitiesFilePath || !probmapDensitiesExists || !probmapDensitiesPath) return;
    setProbmapDensitiesFilePath(probmapDensitiesPath);
  }, [
    densityControlsSelected,
    legacyProbabilityMapSelected,
    probmapDensitiesExists,
    probmapDensitiesPath,
    runDensityEstimation,
    runProbabilityMapStage2,
  ]);

  async function validateInputFolder() {
    if (!inputPath) return;

    const requestId = validationRequestIdRef.current + 1;
    validationRequestIdRef.current = requestId;
    setValidationLoading(true);
    setValidationResult(null);

    try {
      const validationUrl =
        selectedWorkflow === "tracking"
          ? "/api/post-processing/tracking/validate-input"
          : generalSegmentationSelected
            ? "/api/post-processing/general-segmentation/validate-input"
            : selectedWorkflow === "segmentation" || selectedWorkflow === "foreground_probability_map"
              ? "/api/post-processing/segmentation/validate-input"
              : "/api/post-processing/process-features/validate-input";
      const resp = await fetch(validationUrl, {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ path: inputPath }),
      });

      if (!resp.ok) {
        const json = await safeJson(resp);
        const detail =
          json && typeof json === "object" && "detail" in json && typeof json.detail === "string" ? json.detail : null;
        throw new Error(
          detail ? `Validation failed: ${detail}` : `Validation failed: ${resp.status} ${resp.statusText}`
        );
      }

      const json = (await resp.json()) as PostProcessingValidationResult;

      if (requestId !== validationRequestIdRef.current) return;
      setValidationResult(json);
    } catch (error) {
      if (requestId !== validationRequestIdRef.current) return;
      const message = error instanceof Error ? error.message : "Unknown error";
      setValidationResult({ valid: false, reasonCode: "request_failed", message });
    } finally {
      if (requestId === validationRequestIdRef.current) {
        setValidationLoading(false);
      }
    }
  }

  async function validateTrackingSegmentationFolder() {
    if (!trackingSelected || !inputPath || !trackingSegmentationPath) return;

    const requestId = trackingSegmentationValidationRequestIdRef.current + 1;
    trackingSegmentationValidationRequestIdRef.current = requestId;
    setTrackingSegmentationValidationLoading(true);
    setTrackingSegmentationValidationResult(null);

    try {
      const resp = await fetch("/api/post-processing/tracking/validate-segmentation-folder", {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ input_path: inputPath, segmentation_path: trackingSegmentationPath }),
      });

      if (!resp.ok) {
        const json = await safeJson(resp);
        const detail =
          json && typeof json === "object" && "detail" in json && typeof json.detail === "string" ? json.detail : null;
        throw new Error(
          detail ? `Validation failed: ${detail}` : `Validation failed: ${resp.status} ${resp.statusText}`
        );
      }

      const json = (await resp.json()) as PostProcessingValidationResult;
      if (requestId !== trackingSegmentationValidationRequestIdRef.current) return;
      setTrackingSegmentationValidationResult(json);
    } catch (error) {
      if (requestId !== trackingSegmentationValidationRequestIdRef.current) return;
      const message = error instanceof Error ? error.message : "Unknown error";
      setTrackingSegmentationValidationResult({ valid: false, reasonCode: "request_failed", message });
    } finally {
      if (requestId === trackingSegmentationValidationRequestIdRef.current) {
        setTrackingSegmentationValidationLoading(false);
      }
    }
  }

  function buildProcessFeaturesRunRequest(): ProcessFeaturesRunRequest | null {
    if (!inputPath || !effectiveProcessFeaturesOutputPath) return null;

    const parsedPcaComponents = Number.parseInt(pcaComponents.trim(), 10);
    if (pcaSelected && (!Number.isFinite(parsedPcaComponents) || parsedPcaComponents < 1)) {
      return null;
    }
    const parsedTrailingChannels = Number.parseInt(trailingChannels.trim(), 10);
    if (
      featureStatisticsSelected &&
      (!Number.isFinite(parsedTrailingChannels) || parsedTrailingChannels < 0)
    ) {
      return null;
    }

    return {
      input_path: inputPath,
      output_path: effectiveProcessFeaturesOutputPath,
      gpu_index: selectedGpuIndex,
      file_range: {
        start: parseNullableInteger(processFeaturesFileRange.start),
        end: parseNullableInteger(processFeaturesFileRange.end),
      },
      save_high_resolution_features: highResolutionFeaturesSelected,
      high_resolution_save_format: highResolutionSaveFormat,
      save_pca: pcaSelected,
      pca_components: Number.isFinite(parsedPcaComponents) && parsedPcaComponents > 0 ? parsedPcaComponents : 3,
      pca_save_format: pcaSaveFormat,
      global_pca: globalPca,
      save_feature_statistics: featureStatisticsSelected,
      ignore_trailing_channels: ignoreTrailingChannels,
      trailing_channels:
        Number.isFinite(parsedTrailingChannels) && parsedTrailingChannels >= 0 ? parsedTrailingChannels : 6,
    };
  }

  async function submitRun(
    url: string,
    request: ProcessFeaturesRunRequest | ForegroundProbabilityMapRunRequest | SegmentationRunRequest | TrackingRunRequest
  ) {
    setSubmitting(true);
    setRunFeedback(null);

    try {
      const resp = await fetch(url, {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
          "X-SpatialDINO-ClientId": getClientId(),
        },
        body: JSON.stringify(request),
      });

      if (!resp.ok) {
        const json = await safeJson(resp);
        const detail =
          json && typeof json === "object" && "detail" in json && typeof json.detail === "string" ? json.detail : null;
        throw new Error(detail ? detail : `Run failed: ${resp.status} ${resp.statusText}`);
      }

      const json = (await resp.json()) as RunResponse;
      if (json.submitted) {
        setRunFeedback({ tone: "success", message: json.message });
        void jobs.refresh();
        return;
      }

      setRunFeedback({ tone: "error", message: json.message });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      setRunFeedback({ tone: "error", message });
    } finally {
      setSubmitting(false);
    }
  }

  async function handleProcessFeaturesRun() {
    if (!inputPath) {
      setRunFeedback({ tone: "error", message: "Choose an inference output folder." });
      return;
    }
    if (saveProcessFeaturesInDifferentOutputFolder && !processFeaturesOutputPath) {
      setRunFeedback({ tone: "error", message: "Choose an output folder." });
      return;
    }
    const request = buildProcessFeaturesRunRequest();
    if (!request) {
      setRunFeedback({
        tone: "error",
        message: featureStatisticsSelected
          ? "Enter a valid nonnegative integer for trailing channels."
          : "Enter a valid positive integer for the number of PCA components.",
      });
      return;
    }
    if (selectedGpuIndex === null) {
      setRunFeedback({ tone: "error", message: "Select one GPU." });
      return;
    }
    await submitRun("/api/post-processing/process-features/run", request);
  }

  function buildProbabilityMapBaseRequest(outputPath: string | null): ForegroundProbabilityMapRunRequest | null {
    if (!inputPath) {
      setRunFeedback({ tone: "error", message: "Choose an inference output folder." });
      return null;
    }
    if (!outputPath) {
      setRunFeedback({ tone: "error", message: "Choose an output folder." });
      return null;
    }
    if (selectedGpuIndex === null) {
      setRunFeedback({ tone: "error", message: "Select one GPU." });
      return null;
    }

    const selectedTrainingTimepoint =
      validatedSubfolderNames.length === 1 ? validatedSubfolderNames[0] ?? "" : densityEstimationTimepoint;

    const parsedFeatureBatch = Number.parseInt(probabilityMapFeatureBatch.trim(), 10);
    if (!Number.isFinite(parsedFeatureBatch) || parsedFeatureBatch < 1) {
      setRunFeedback({ tone: "error", message: "Enter a valid positive integer for Feature batch." });
      return null;
    }

    const parsedKdePoints = Number.parseInt(probabilityMapKdePoints.trim(), 10);
    if (!Number.isFinite(parsedKdePoints) || parsedKdePoints < 2) {
      setRunFeedback({ tone: "error", message: "Enter a valid integer of at least 2 for Density grid size." });
      return null;
    }

    const parsedKdeMaxSamples = Number.parseInt(probabilityMapKdeMaxSamples.trim(), 10);
    if (!Number.isFinite(parsedKdeMaxSamples) || parsedKdeMaxSamples < 1) {
      setRunFeedback({ tone: "error", message: "Enter a valid positive integer for Max samples per class." });
      return null;
    }

    const parsedSeed = Number.parseInt(probabilityMapSeed.trim(), 10);
    if (!Number.isFinite(parsedSeed)) {
      setRunFeedback({ tone: "error", message: "Enter a valid integer for Random seed." });
      return null;
    }

    const parsedHistSigmaBins = Number.parseFloat(probabilityMapHistSigmaBins.trim());
    if (
      probabilityMapDensityMethod === "gpu-hist" &&
      (!Number.isFinite(parsedHistSigmaBins) || parsedHistSigmaBins <= 0)
    ) {
      setRunFeedback({ tone: "error", message: "Enter a valid positive number for Histogram sigma." });
      return null;
    }

    let parsedKdeBandwidth: number | null = null;
    if (probabilityMapDensityMethod === "kde" && probabilityMapKdeBandwidth.trim()) {
      parsedKdeBandwidth = Number.parseFloat(probabilityMapKdeBandwidth.trim());
      if (!Number.isFinite(parsedKdeBandwidth) || parsedKdeBandwidth <= 0) {
        setRunFeedback({ tone: "error", message: "Enter a valid positive KDE bandwidth, or leave it blank for auto." });
        return null;
      }
    }

    if (runDensityEstimation) {
      if (!selectedTrainingTimepoint) {
        setRunFeedback({ tone: "error", message: "Choose one training timepoint for Stage 1." });
        return null;
      }
      if (!segTifPath) {
        setRunFeedback({ tone: "error", message: "Choose an annotated FG/BG mask for Stage 1." });
        return null;
      }
    } else if (!probmapDensitiesFilePath) {
      setRunFeedback({
        tone: "error",
        message: "Choose a Stage 1 output file.",
      });
      return null;
    }

    return {
      input_path: inputPath,
      output_path: outputPath,
      densities_path: runDensityEstimation ? null : probmapDensitiesFilePath,
      gpu_index: selectedGpuIndex,
      run_density_estimation: runDensityEstimation,
      training_timepoint: runDensityEstimation ? selectedTrainingTimepoint : null,
      seg_tif: runDensityEstimation ? segTifPath : null,
      valid_mask_tif: runDensityEstimation ? validMaskTifPath : null,
      density_method: probabilityMapDensityMethod,
      feature_batch: parsedFeatureBatch,
      kde_points: parsedKdePoints,
      kde_max_samples: parsedKdeMaxSamples,
      kde_bandwidth: probabilityMapDensityMethod === "kde" ? parsedKdeBandwidth : null,
      hist_sigma_bins: probabilityMapDensityMethod === "gpu-hist" ? parsedHistSigmaBins : 1.5,
      seed: parsedSeed,
    };
  }

  async function handleForegroundProbabilityMapRun() {
    if (saveSegmentationInDifferentOutputFolder && !segmentationOutputPath) {
      setRunFeedback({ tone: "error", message: "Choose an output folder." });
      return;
    }

    const request = buildProbabilityMapBaseRequest(effectiveSegmentationOutputPath);
    if (!request) return;
    await submitRun("/api/post-processing/foreground-probability-map/run", request);
  }

  async function handleSegmentationRun() {
    if (!inputPath) {
      setRunFeedback({ tone: "error", message: "Choose an inference output folder." });
      return;
    }

    if (saveSegmentationInDifferentOutputFolder && !segmentationOutputPath) {
      setRunFeedback({ tone: "error", message: "Choose a segmentation output folder." });
      return;
    }

    if (selectedSegmentationMode === "general_segmentation") {
      if (!selectedGeneralSegmentationSource) {
        setRunFeedback({ tone: "error", message: "Choose a segmentation source." });
        return;
      }

      const parsedThreshold = Number.parseFloat(generalSegmentationThreshold.trim());
      if (!Number.isFinite(parsedThreshold)) {
        setRunFeedback({ tone: "error", message: "Enter a valid threshold." });
        return;
      }

      const dataOperationsResult = serializeGeneralDataOperations(
        generalSegmentationProcessData,
        generalSegmentationDataOperations,
        true
      );
      if (dataOperationsResult.error) {
        setRunFeedback({ tone: "error", message: dataOperationsResult.error });
        return;
      }
      const maskOperationsResult = serializeGeneralMaskOperations(
        generalSegmentationProcessMask,
        generalSegmentationMaskOperations,
        true
      );
      if (maskOperationsResult.error) {
        setRunFeedback({ tone: "error", message: maskOperationsResult.error });
        return;
      }
      const parsedVoronoiSpotSigma = parseNonnegativeNumber(generalSegmentationVoronoiSpotSigma);
      if (parsedVoronoiSpotSigma === null) {
        setRunFeedback({ tone: "error", message: "Enter a valid nonnegative Voronoi-Otsu spot sigma." });
        return;
      }
      const parsedVoronoiOutlineSigma = parseNonnegativeNumber(generalSegmentationVoronoiOutlineSigma);
      if (parsedVoronoiOutlineSigma === null) {
        setRunFeedback({ tone: "error", message: "Enter a valid nonnegative Voronoi-Otsu outline sigma." });
        return;
      }
      const parsedDistanceDynamic = parseNonnegativeNumber(generalSegmentationDistanceDynamic);
      if (parsedDistanceDynamic === null) {
        setRunFeedback({ tone: "error", message: "Enter a valid nonnegative distance-transform dynamic value." });
        return;
      }
      const parsedDistanceSpacingZ = parsePositiveNumber(generalSegmentationDistanceSpacingZ);
      const parsedDistanceSpacingY = parsePositiveNumber(generalSegmentationDistanceSpacingY);
      const parsedDistanceSpacingX = parsePositiveNumber(generalSegmentationDistanceSpacingX);
      if (parsedDistanceSpacingZ === null || parsedDistanceSpacingY === null || parsedDistanceSpacingX === null) {
        setRunFeedback({ tone: "error", message: "Enter valid positive distance-transform spacing values." });
        return;
      }
      const parsedIntensityProminence = parseNonnegativeNumber(generalSegmentationIntensityProminence);
      if (parsedIntensityProminence === null || parsedIntensityProminence > 1) {
        setRunFeedback({ tone: "error", message: "Enter an intensity prominence between 0 and 1." });
        return;
      }
      const parsedIntensitySmoothingSigma = parseNonnegativeNumber(generalSegmentationIntensitySmoothingSigma);
      if (parsedIntensitySmoothingSigma === null) {
        setRunFeedback({ tone: "error", message: "Enter a valid nonnegative intensity smoothing sigma." });
        return;
      }
      const parsedIntensityLowPercentile = parseNonnegativeNumber(generalSegmentationIntensityLowPercentile);
      const parsedIntensityHighPercentile = parseNonnegativeNumber(generalSegmentationIntensityHighPercentile);
      if (
        parsedIntensityLowPercentile === null ||
        parsedIntensityHighPercentile === null ||
        parsedIntensityLowPercentile > 100 ||
        parsedIntensityHighPercentile > 100 ||
        parsedIntensityLowPercentile >= parsedIntensityHighPercentile
      ) {
        setRunFeedback({ tone: "error", message: "Enter ordered intensity normalization percentiles between 0 and 100." });
        return;
      }

      const maxComponent = Math.max(0, selectedGeneralSegmentationSource.componentCount - 1);
      const thresholdComponent = clampInteger(generalSegmentationThresholdComponent, 0, maxComponent);

      await submitRun("/api/post-processing/segmentation/run", {
        input_path: inputPath,
        output_path: effectiveSegmentationOutputPath,
        densities_path: null,
        gpu_index: selectedGpuIndex,
        mode: "general_segmentation",
        source_id: selectedGeneralSegmentationSource.id,
        threshold: parsedThreshold,
        component_index: thresholdComponent,
        invert_mask: generalSegmentationInvertMask,
        data_operations: dataOperationsResult.operations,
        mask_operations: maskOperationsResult.operations,
        instance_method: generalSegmentationInstanceMethod,
        voronoi_spot_sigma: parsedVoronoiSpotSigma,
        voronoi_outline_sigma: parsedVoronoiOutlineSigma,
        distance_transform_dynamic: parsedDistanceDynamic,
        distance_transform_connectivity: generalSegmentationDistanceConnectivity,
        distance_transform_spacing_z: parsedDistanceSpacingZ,
        distance_transform_spacing_y: parsedDistanceSpacingY,
        distance_transform_spacing_x: parsedDistanceSpacingX,
        intensity_prominence: parsedIntensityProminence,
        intensity_smoothing_sigma: parsedIntensitySmoothingSigma,
        intensity_low_percentile: parsedIntensityLowPercentile,
        intensity_high_percentile: parsedIntensityHighPercentile,
        intensity_connectivity: generalSegmentationIntensityConnectivity,
        gaussian_blur_sigma: 3,
        rolling_ball_radius: 10,
        run_density_estimation: false,
        run_stage_2: true,
        training_timepoint: null,
        seg_tif: null,
        valid_mask_tif: null,
        density_method: "gpu-hist",
        feature_batch: 32,
        kde_points: 512,
        kde_max_samples: 200000,
        kde_bandwidth: null,
        hist_sigma_bins: 1.5,
        bg_prob_threshold: 0.4,
        fg_prob_threshold: 0.95,
        probmap_threshold: selectedGeneralSegmentationSource.kind === "probmap" ? parsedThreshold : 0.5,
        run_connected_components: generalSegmentationInstanceMethod === "connected_components",
        seed: 1337,
      });
      return;
    }

    if (selectedGpuIndex === null) {
      setRunFeedback({ tone: "error", message: "Select one GPU." });
      return;
    }

    if (selectedSegmentationMode === "voronoi_otsu") {
      const parsedGaussianBlurSigma = Number.parseInt(gaussianBlurSigma.trim(), 10);
      if (!Number.isFinite(parsedGaussianBlurSigma) || parsedGaussianBlurSigma < 0) {
        setRunFeedback({ tone: "error", message: "Enter a valid nonnegative integer for Gaussian blur sigma." });
        return;
      }

      const parsedRollingBallRadius = Number.parseFloat(rollingBallRadius.trim());
      if (!Number.isFinite(parsedRollingBallRadius) || parsedRollingBallRadius < 0) {
        setRunFeedback({ tone: "error", message: "Enter a valid nonnegative number for Rolling ball radius." });
        return;
      }

      await submitRun("/api/post-processing/segmentation/run", {
        input_path: inputPath,
        output_path: effectiveSegmentationOutputPath,
        densities_path: null,
        gpu_index: selectedGpuIndex,
        mode: "voronoi_otsu",
        source_id: "raw",
        threshold: null,
        component_index: 0,
        invert_mask: false,
        data_operations: [],
        mask_operations: [],
        instance_method: "connected_components",
        voronoi_spot_sigma: 2,
        voronoi_outline_sigma: 2,
        gaussian_blur_sigma: parsedGaussianBlurSigma,
        rolling_ball_radius: parsedRollingBallRadius,
        run_density_estimation: false,
        run_stage_2: true,
        training_timepoint: null,
        seg_tif: null,
        valid_mask_tif: null,
        density_method: "gpu-hist",
        feature_batch: 32,
        kde_points: 512,
        kde_max_samples: 200000,
        kde_bandwidth: null,
        hist_sigma_bins: 1.5,
        bg_prob_threshold: 0.4,
        fg_prob_threshold: 0.95,
        probmap_threshold: 0.5,
        run_connected_components: true,
        seed: 1337,
      });
      return;
    }

    if (!runDensityEstimation && !runProbabilityMapStage2) {
      setRunFeedback({ tone: "error", message: "Choose at least one stage: Run stage 1 and/or Run stage 2." });
      return;
    }

    const baseRequest = buildProbabilityMapBaseRequest(effectiveSegmentationOutputPath);
    if (!baseRequest) return;

    const parsedBgThreshold = Number.parseFloat(probabilityMapBgThreshold.trim());
    if (!Number.isFinite(parsedBgThreshold) || parsedBgThreshold < 0 || parsedBgThreshold > 1) {
      setRunFeedback({ tone: "error", message: "Enter a valid background probability threshold between 0 and 1." });
      return;
    }

    const parsedFgThreshold = Number.parseFloat(probabilityMapFgThreshold.trim());
    if (!Number.isFinite(parsedFgThreshold) || parsedFgThreshold < 0 || parsedFgThreshold > 1) {
      setRunFeedback({ tone: "error", message: "Enter a valid foreground probability threshold between 0 and 1." });
      return;
    }

    await submitRun("/api/post-processing/segmentation/run", {
      ...baseRequest,
      mode: "legacy_probability_map",
      source_id: "raw",
      threshold: null,
      component_index: 0,
      invert_mask: false,
      data_operations: [],
      mask_operations: [],
      instance_method: "connected_components",
      voronoi_spot_sigma: 2,
      voronoi_outline_sigma: 2,
      gaussian_blur_sigma: 3,
      rolling_ball_radius: 10,
      run_stage_2: runProbabilityMapStage2,
      bg_prob_threshold: parsedBgThreshold,
      fg_prob_threshold: parsedFgThreshold,
      probmap_threshold: 0.5,
      run_connected_components: true,
    });
  }

  async function handleTrackingRun() {
    if (!inputPath) {
      setRunFeedback({ tone: "error", message: "Choose an inference output folder." });
      return;
    }
    if (!trackingSegmentationPath) {
      setRunFeedback({ tone: "error", message: "Choose a segmentation output folder." });
      return;
    }
    if (saveTrackingInDifferentOutputFolder && !trackingOutputPath) {
      setRunFeedback({ tone: "error", message: "Choose an output folder." });
      return;
    }

    const trimmedOutputFilename = trackingOutputFilename.trim();
    if (!trimmedOutputFilename) {
      setRunFeedback({ tone: "error", message: "Enter an output file name." });
      return;
    }
    if (
      trimmedOutputFilename === "." ||
      trimmedOutputFilename === ".." ||
      trimmedOutputFilename.includes("/") ||
      trimmedOutputFilename.includes("\\")
    ) {
      setRunFeedback({ tone: "error", message: "Output file name must be a file name, not a path." });
      return;
    }
    let normalizedOutputFilename = trimmedOutputFilename;
    if (!trimmedOutputFilename.includes(".")) {
      normalizedOutputFilename = `${trimmedOutputFilename}.csv`;
      setTrackingOutputFilename(normalizedOutputFilename);
    } else if (!trimmedOutputFilename.toLowerCase().endsWith(".csv")) {
      setRunFeedback({ tone: "error", message: "Output file name must end in .csv." });
      return;
    }

    const parsedMaxDistanceXy = Number.parseFloat(trackingMaxDistanceXy.trim());
    if (!Number.isFinite(parsedMaxDistanceXy) || parsedMaxDistanceXy <= 0) {
      setRunFeedback({ tone: "error", message: "Enter a valid positive number for Max XY distance." });
      return;
    }

    const parsedMaxDistanceZ = Number.parseFloat(trackingMaxDistanceZ.trim());
    if (!Number.isFinite(parsedMaxDistanceZ) || parsedMaxDistanceZ <= 0) {
      setRunFeedback({ tone: "error", message: "Enter a valid positive number for Max Z distance." });
      return;
    }

    const parsedZDistanceWeight = Number.parseFloat(trackingZDistanceWeight.trim());
    if (!Number.isFinite(parsedZDistanceWeight) || parsedZDistanceWeight <= 0) {
      setRunFeedback({ tone: "error", message: "Enter a valid positive number for Z distance weight." });
      return;
    }

    const parsedMinDistanceToRemoveCand = Number.parseFloat(trackingMinDistanceToRemoveCand.trim());
    if (!Number.isFinite(parsedMinDistanceToRemoveCand) || parsedMinDistanceToRemoveCand < 0) {
      setRunFeedback({ tone: "error", message: "Enter a valid nonnegative immediate assignment distance." });
      return;
    }

    const parsedDiceThreshold = Number.parseFloat(trackingDiceThreshold.trim());
    if (!Number.isFinite(parsedDiceThreshold) || parsedDiceThreshold < 0 || parsedDiceThreshold > 1) {
      setRunFeedback({ tone: "error", message: "Enter a valid Dice threshold between 0 and 1." });
      return;
    }

    const parsedCorrThreshold = Number.parseFloat(trackingCorrThreshold.trim());
    if (!Number.isFinite(parsedCorrThreshold) || parsedCorrThreshold < -1 || parsedCorrThreshold > 1) {
      setRunFeedback({ tone: "error", message: "Enter a valid correlation threshold between -1 and 1." });
      return;
    }

    const normalizedVoteThresholds = trackingVoteThresholds
      .split(",")
      .map((token) => token.trim())
      .filter((token) => token.length > 0);
    if (normalizedVoteThresholds.some((token) => !/^\d+$/.test(token) || Number.parseInt(token, 10) <= 0)) {
      setRunFeedback({ tone: "error", message: "Vote thresholds must be a comma-separated list of positive minimum counts." });
      return;
    }

    const trimmedMinFeatureVotes = trackingMinFeatureVotes.trim();
    const parsedMinFeatureVotes = Number.parseInt(trimmedMinFeatureVotes, 10);
    if (!/^\d+$/.test(trimmedMinFeatureVotes) || parsedMinFeatureVotes <= 0) {
      setRunFeedback({ tone: "error", message: "Enter a valid positive integer for Min feature votes." });
      return;
    }

    await submitRun("/api/post-processing/tracking/run", {
      input_path: inputPath,
      segmentation_path: trackingSegmentationPath,
      output_path: effectiveTrackingOutputPath ?? inputPath,
      output_filename: normalizedOutputFilename,
      max_distance_xy: parsedMaxDistanceXy,
      max_distance_z: parsedMaxDistanceZ,
      z_distance_weight: parsedZDistanceWeight,
      min_distance_to_remove_cand: parsedMinDistanceToRemoveCand,
      vote_thresholds: normalizedVoteThresholds.join(","),
      dice_threshold: parsedDiceThreshold,
      corr_threshold: parsedCorrThreshold,
      save_extended_results: trackingSaveExtendedResults,
      ignore_features: trackingIgnoreFeatures,
      disable_centroid_fallback: trackingDisableCentroidFallback,
      aggressive_feature_matching: trackingAggressiveFeatureMatching,
      min_feature_votes: parsedMinFeatureVotes,
    });
  }

  return (
    <div className="preprocessPage">
      <section className="validationCard inferenceIntroCard" aria-label={`${pageTitle} overview`}>
        <header className="validationHeader">
          <div>
            <h1 className="inferenceTitle">{pageTitle}</h1>
          </div>
        </header>
      </section>

      {pageKind === "post_processing" ? (
        <section className="datasetCard" aria-label="Post-processing options">
          <div className="postProcessingOptionGrid">
            {POST_PROCESSING_OPTIONS.map((option) => {
              const active = option.value === selectedPostProcessingMode;
              return (
                <button
                  key={option.value}
                  type="button"
                  className={active ? "postProcessingOptionButton isActive" : "postProcessingOptionButton"}
                  aria-pressed={active}
                  onClick={() => setSelectedPostProcessingMode(option.value)}
                >
                  <div className="postProcessingOptionLabel">{option.label}</div>
                </button>
              );
            })}
          </div>
        </section>
      ) : null}

      {segmentationSelected ? (
        <section className="datasetCard" aria-label="Segmentation workflows">
          <div className="segmentationOptionGrid">
            {SEGMENTATION_OPTIONS.map((option) => {
              const active = option.value === selectedSegmentationMode;
              return (
                <button
                  key={option.value}
                  type="button"
                  className={active ? "postProcessingOptionButton isActive" : "postProcessingOptionButton"}
                  aria-pressed={active}
                  onClick={() => setSelectedSegmentationMode(option.value)}
                >
                  <div className="postProcessingOptionLabel">{option.label}</div>
                </button>
              );
            })}
          </div>
        </section>
      ) : null}

      <section className="datasetCard inferenceInputCard" aria-label={`${pageTitle} folder selection`}>
          <DirectoryFieldRow
            label={
              <ParameterHelpLabel
                label="Inference output folder"
                description={POST_PROCESSING_PARAMETER_HELP.inferenceOutputFolder}
              />
            }
            path={inputPath}
            buttonLabel="Choose directory"
            emptyLabel="No directory selected yet"
            onChoose={() => {
              setPickerTarget("input");
              setPickerOpen(true);
            }}
            action={
              <button
                type="button"
                className="preprocessValidateButton"
                disabled={!inputPath || validationLoading}
                onClick={() => void validateInputFolder()}
              >
                {validationLoading ? "Validating..." : "Validate"}
              </button>
            }
          />

          {validationLoading ? (
            <ValidationMessage tone="neutral">Validating inference output folder...</ValidationMessage>
          ) : validationResult?.valid && inputValidationSuccessMessage ? (
            <ValidationMessage tone="success">{inputValidationSuccessMessage}</ValidationMessage>
          ) : validationResult ? (
            <ValidationMessage tone="error">{validationResult.message}</ValidationMessage>
          ) : null}

          {processFeaturesSelected ? (
            <>
              <div className="inferenceFormRow">
                <div className="inferenceFieldLabel">
                  <ParameterHelpLabel
                    label="Save in a different output folder"
                    description={POST_PROCESSING_PARAMETER_HELP.processFeaturesOutputFolder}
                  />
                </div>
                <label className="inferenceCheckboxLabel">
                  <input
                    type="checkbox"
                    checked={saveProcessFeaturesInDifferentOutputFolder}
                    onChange={(event) => setSaveProcessFeaturesInDifferentOutputFolder(event.target.checked)}
                  />
                  <span>Enabled</span>
                </label>
              </div>

              {saveProcessFeaturesInDifferentOutputFolder ? (
                <DirectoryFieldRow
                  label={
                    <ParameterHelpLabel
                      label="Output folder"
                      description={POST_PROCESSING_PARAMETER_HELP.processFeaturesOutputFolder}
                    />
                  }
                  path={processFeaturesOutputPath}
                  buttonLabel="Choose directory"
                  emptyLabel="No directory selected yet"
                  onChoose={() => {
                    setPickerTarget("process_output");
                    setPickerOpen(true);
                  }}
                />
              ) : null}
            </>
          ) : null}

          {foregroundProbabilityMapSelected ? (
            <>
              <div className="inferenceFormRow">
                <div className="inferenceFieldLabel">
                  <ParameterHelpLabel
                    label="Save in a different output folder"
                    description={POST_PROCESSING_PARAMETER_HELP.foregroundProbabilityMapOutputFolder}
                  />
                </div>
                <label className="inferenceCheckboxLabel">
                  <input
                    type="checkbox"
                    checked={saveSegmentationInDifferentOutputFolder}
                    onChange={(event) => setSaveSegmentationInDifferentOutputFolder(event.target.checked)}
                  />
                  <span>Enabled</span>
                </label>
              </div>

              {saveSegmentationInDifferentOutputFolder ? (
                <DirectoryFieldRow
                  label={
                    <ParameterHelpLabel
                      label="Output folder"
                      description={POST_PROCESSING_PARAMETER_HELP.foregroundProbabilityMapOutputFolder}
                    />
                  }
                  path={segmentationOutputPath}
                  buttonLabel="Choose directory"
                  emptyLabel="No directory selected yet"
                  onChoose={() => {
                    setPickerTarget("segmentation_output");
                    setPickerOpen(true);
                  }}
                />
              ) : null}
            </>
          ) : null}

          {segmentationSelected ? (
            <>
              <div className="inferenceFormRow">
                <div className="inferenceFieldLabel">
                  <ParameterHelpLabel
                    label="Save in a different output folder"
                    description={POST_PROCESSING_PARAMETER_HELP.segmentationOutputFolder}
                  />
                </div>
                <label className="inferenceCheckboxLabel">
                  <input
                    type="checkbox"
                    checked={saveSegmentationInDifferentOutputFolder}
                    onChange={(event) => setSaveSegmentationInDifferentOutputFolder(event.target.checked)}
                  />
                  <span>Enabled</span>
                </label>
              </div>

              {saveSegmentationInDifferentOutputFolder ? (
                <DirectoryFieldRow
                  label={
                    <ParameterHelpLabel
                      label="Segmentation output folder"
                      description={POST_PROCESSING_PARAMETER_HELP.segmentationOutputFolder}
                    />
                  }
                  path={segmentationOutputPath}
                  buttonLabel="Choose directory"
                  emptyLabel="No directory selected yet"
                  onChoose={() => {
                    setPickerTarget("segmentation_output");
                    setPickerOpen(true);
                  }}
                />
              ) : null}
            </>
          ) : null}

          {trackingSelected ? (
            <>
              <DirectoryFieldRow
                label={
                  <ParameterHelpLabel
                    label="Segmentation output folder"
                    description={POST_PROCESSING_PARAMETER_HELP.segmentationFolder}
                  />
                }
                path={trackingSegmentationPath}
                buttonLabel="Choose directory"
                emptyLabel="No directory selected yet"
                onChoose={() => {
                  setPickerTarget("tracking_segmentation");
                  setPickerOpen(true);
                }}
                action={
                  <button
                    type="button"
                    className="preprocessValidateButton"
                    disabled={!inputPath || !trackingSegmentationPath || trackingSegmentationValidationLoading}
                    onClick={() => void validateTrackingSegmentationFolder()}
                  >
                    {trackingSegmentationValidationLoading ? "Validating..." : "Validate"}
                  </button>
                }
              />

              <div className="inferenceFormRow">
                <div className="inferenceFieldLabel">
                  <ParameterHelpLabel
                    label="Save in a different output folder"
                    description={POST_PROCESSING_PARAMETER_HELP.trackingOutputFolder}
                  />
                </div>
                <label className="inferenceCheckboxLabel">
                  <input
                    type="checkbox"
                    checked={saveTrackingInDifferentOutputFolder}
                    onChange={(event) => setSaveTrackingInDifferentOutputFolder(event.target.checked)}
                  />
                  <span>Enabled</span>
                </label>
              </div>

              <div className="inferenceFormRow">
                <div className="inferenceFieldLabel">
                  <ParameterHelpLabel
                    label="File name:"
                    description={POST_PROCESSING_PARAMETER_HELP.trackingOutputFilename}
                  />
                </div>
                <PostProcessingTextInput
                  value={trackingOutputFilename}
                  onChange={setTrackingOutputFilename}
                  className="inferenceTextInputWide"
                />
              </div>

              {saveTrackingInDifferentOutputFolder ? (
                <DirectoryFieldRow
                  label={
                    <ParameterHelpLabel label="Output folder" description={POST_PROCESSING_PARAMETER_HELP.trackingOutputFolder} />
                  }
                  path={trackingOutputPath}
                  buttonLabel="Choose directory"
                  emptyLabel="No directory selected yet"
                  onChoose={() => {
                    setPickerTarget("tracking_output");
                    setPickerOpen(true);
                  }}
                />
              ) : null}

              {trackingSegmentationValidationLoading ? (
                <ValidationMessage tone="neutral">Validating segmentation output folder...</ValidationMessage>
              ) : trackingSegmentationValidationResult?.valid && trackingSegmentationValidationSuccessMessage ? (
                <ValidationMessage tone="success">{trackingSegmentationValidationSuccessMessage}</ValidationMessage>
              ) : trackingSegmentationValidationResult ? (
                <ValidationMessage tone="error">{trackingSegmentationValidationResult.message}</ValidationMessage>
              ) : null}

              {parametersVisible ? (
                <div className="inferenceOptionalToggleRow">
                  <button
                    type="button"
                    className="pickerPrimaryButton"
                    aria-expanded={trackingOptionalOpen}
                    onClick={() => setTrackingOptionalOpen((current) => !current)}
                  >
                    Advanced options
                  </button>
                </div>
              ) : null}
            </>
          ) : null}
      </section>

      {parametersVisible && processFeaturesSelected ? (
        <section className="datasetCard inferenceFormCard" aria-label="Process features parameters">
          {optionsError ? <div className="sidebarError">{optionsError}</div> : null}

          <div className="inferenceFormRows">
            <GpuSelectionRow
              optionsLoading={optionsLoading}
              availableGpus={availableGpus}
              gpuError={gpuError}
              selectedGpuIndex={selectedGpuIndex}
              onSelectGpu={setSelectedGpuIndex}
              helpDescription={POST_PROCESSING_PARAMETER_HELP.selectGpu}
            />

            <div className="inferenceFormRow">
              <div className="inferenceFieldLabel isStrong">
                <ParameterHelpLabel label="Chosen files" description={POST_PROCESSING_PARAMETER_HELP.chosenFiles} />
              </div>
              <div className="inferenceInlineLabel">Start file:</div>
              <PostProcessingNumberInput
                value={processFeaturesFileRange.start}
                onChange={(value) => setProcessFeaturesFileRange((current) => ({ ...current, start: value }))}
                min={0}
                step={1}
                max={maxProcessFeaturesFileIndex}
                ariaLabel="Start file"
              />
              <div className="inferenceInlineLabel">End file:</div>
              <PostProcessingNumberInput
                value={processFeaturesFileRange.end}
                onChange={(value) => setProcessFeaturesFileRange((current) => ({ ...current, end: value }))}
                min={0}
                step={1}
                max={maxProcessFeaturesFileIndex}
                ariaLabel="End file"
              />
            </div>

            {highResolutionFeaturesSelected ? (
              <div className="inferenceFormRow">
                <div className="inferenceFieldLabel">
                  <ParameterHelpLabel label="Save format" description={POST_PROCESSING_PARAMETER_HELP.saveFormat} />
                </div>
                <select
                  className="inferenceSelect inferenceCompactSelect"
                  value={highResolutionSaveFormat}
                  onChange={(event) => setHighResolutionSaveFormat(event.target.value as SaveFormat)}
                >
                  <option value=".npy">.npy</option>
                  <option value=".tif">.tif</option>
                </select>
              </div>
            ) : null}

            {highResolutionFeaturesSelected ? (
              <div className="sidebarWarning">
                Saving high-resolution features writes one full 3D file per feature, typically 390 files per
                timepoint, and can consume a huge amount of disk space.
              </div>
            ) : null}

            {pcaSelected ? (
              <div className="inferenceFormRow">
                <div className="inferenceFieldLabel">
                  <ParameterHelpLabel label="Components" description={POST_PROCESSING_PARAMETER_HELP.components} />
                </div>
                <PostProcessingNumberInput value={pcaComponents} onChange={setPcaComponents} min={1} step={1} />
                <div className="inferenceInlineLabel isStrong">
                  <ParameterHelpLabel label="Save format" description={POST_PROCESSING_PARAMETER_HELP.saveFormat} />
                </div>
                <select
                  className="inferenceSelect inferenceCompactSelect postProcessingPcaFormatSelect"
                  value={pcaSaveFormat}
                  onChange={(event) => setPcaSaveFormat(event.target.value as SaveFormat)}
                >
                  <option value=".npy">.npy</option>
                  <option value=".tif">.tif</option>
                </select>
                <label className="inferenceCheckboxLabel">
                  <input
                    type="checkbox"
                    checked={globalPca}
                    onChange={(event) => setGlobalPca(event.target.checked)}
                  />
                  <span>
                    <ParameterHelpLabel label="Global PCA" description={POST_PROCESSING_PARAMETER_HELP.globalPca} />
                  </span>
                </label>
              </div>
            ) : null}

            {featureStatisticsSelected ? (
              <>
                <div className="inferenceFormRow">
                  <div className="inferenceFieldLabel">
                    <ParameterHelpLabel
                      label="Feature statistics"
                      description={POST_PROCESSING_PARAMETER_HELP.featureStatistics}
                    />
                  </div>
                  <label className="inferenceCheckboxLabel">
                    <input
                      type="checkbox"
                      checked={ignoreTrailingChannels}
                      onChange={(event) => setIgnoreTrailingChannels(event.target.checked)}
                    />
                    <span>
                      <ParameterHelpLabel
                        label="Ignore trailing channels"
                        description={POST_PROCESSING_PARAMETER_HELP.ignoreTrailingChannels}
                      />
                    </span>
                  </label>
                  <div className="inferenceInlineLabel isStrong">
                    <ParameterHelpLabel
                      label="Trailing channels"
                      description={POST_PROCESSING_PARAMETER_HELP.trailingChannels}
                    />
                  </div>
                  <PostProcessingNumberInput value={trailingChannels} onChange={setTrailingChannels} min={0} step={1} />
                </div>
                <div className="sidebarHint">
                  Writes mean, max, min, median, standard deviation, and L2 norm volumes under{" "}
                  <code>feature_stats/</code>.
                </div>
              </>
            ) : null}
          </div>

          <div className="validationActions">
            <button
              type="button"
              className="preprocessValidateButton"
              onClick={() => void handleProcessFeaturesRun()}
              disabled={submitting}
            >
              {submitting ? "Submitting..." : "Run"}
            </button>
          </div>

          {runFeedback ? <ValidationMessage tone={runFeedback.tone}>{runFeedback.message}</ValidationMessage> : null}
        </section>
      ) : null}

      {parametersVisible && (segmentationSelected || foregroundProbabilityMapSelected) ? (
        <section className="datasetCard inferenceFormCard" aria-label={`${pageTitle} parameters`}>
          {optionsError ? <div className="sidebarError">{optionsError}</div> : null}

          {voronoiOtsuSelected ? (
            <div className="inferenceFormRows">
              <GpuSelectionRow
                optionsLoading={optionsLoading}
                availableGpus={availableGpus}
                gpuError={gpuError}
                selectedGpuIndex={selectedGpuIndex}
                onSelectGpu={setSelectedGpuIndex}
                helpDescription={POST_PROCESSING_PARAMETER_HELP.selectGpu}
              />

              <div className="inferenceOptionalToggleRow">
                <button
                  type="button"
                  className="pickerPrimaryButton"
                  aria-expanded={voronoiOptionalOpen}
                  aria-controls="voronoi-optional-parameters"
                  onClick={() => setVoronoiOptionalOpen((current) => !current)}
                >
                  Advanced options
                </button>
              </div>

              {voronoiOptionalOpen ? (
                <div id="voronoi-optional-parameters" className="inferenceOptionalSection">
                  <div className="inferenceFormRow">
                    <div className="inferenceFieldLabel">
                      <ParameterHelpLabel
                        label="Gaussian blur sigma"
                        description={POST_PROCESSING_PARAMETER_HELP.gaussianBlurSigma}
                      />
                    </div>
                    <PostProcessingNumberInput
                      value={gaussianBlurSigma}
                      onChange={setGaussianBlurSigma}
                      min={0}
                      step={1}
                    />
                    <div className="inferenceInlineLabel isStrong">
                      <ParameterHelpLabel
                        label="Rolling ball radius"
                        description={POST_PROCESSING_PARAMETER_HELP.rollingBallRadius}
                      />
                    </div>
                    <PostProcessingNumberInput
                      value={rollingBallRadius}
                      onChange={setRollingBallRadius}
                      min={0}
                      step={0.1}
                    />
                  </div>
                </div>
              ) : null}
            </div>
          ) : null}

          {generalSegmentationSelected ? (
            <div className="inferenceFormRows">
              {generalSegmentationSourceWarnings.map((warning) => (
                <div key={warning.folderName} className="sidebarError">
                  {`${warning.folderName}/ was found but is not valid: ${warning.message}`}
                </div>
              ))}

              <GpuSelectionRow
                optionsLoading={optionsLoading}
                availableGpus={availableGpus}
                gpuError={gpuError}
                selectedGpuIndex={selectedGpuIndex}
                onSelectGpu={setSelectedGpuIndex}
                helpDescription={POST_PROCESSING_PARAMETER_HELP.selectGpu}
              />

              <GeneralSegmentationPreview
                inputPath={inputPath}
                gpuIndex={selectedGpuIndex}
                enabled={generalSegmentationSelected && parametersVisible && Boolean(selectedGeneralSegmentationSource)}
                sources={generalSegmentationSources}
                source={selectedGeneralSegmentationSource}
                sourceId={generalSegmentationSourceId}
                onSourceIdChange={(sourceId) => {
                  setGeneralSegmentationSourceId(sourceId);
                  setGeneralSegmentationThresholdComponent(0);
                }}
                thresholdComponent={generalSegmentationThresholdComponent}
                onThresholdComponentChange={setGeneralSegmentationThresholdComponent}
                displaySource={generalSegmentationDisplaySource}
                onDisplaySourceChange={setGeneralSegmentationDisplaySource}
                processData={generalSegmentationProcessData}
                onProcessDataChange={setGeneralSegmentationProcessData}
                dataOperations={generalSegmentationDataOperations}
                onDataOperationsChange={setGeneralSegmentationDataOperations}
                threshold={generalSegmentationThreshold}
                onThresholdChange={setGeneralSegmentationThreshold}
                invertMask={generalSegmentationInvertMask}
                onInvertMaskChange={setGeneralSegmentationInvertMask}
                processMask={generalSegmentationProcessMask}
                onProcessMaskChange={setGeneralSegmentationProcessMask}
                maskOperations={generalSegmentationMaskOperations}
                onMaskOperationsChange={setGeneralSegmentationMaskOperations}
                instanceMethod={generalSegmentationInstanceMethod}
                onInstanceMethodChange={setGeneralSegmentationInstanceMethod}
                voronoiSpotSigma={generalSegmentationVoronoiSpotSigma}
                onVoronoiSpotSigmaChange={setGeneralSegmentationVoronoiSpotSigma}
                voronoiOutlineSigma={generalSegmentationVoronoiOutlineSigma}
                onVoronoiOutlineSigmaChange={setGeneralSegmentationVoronoiOutlineSigma}
                distanceDynamic={generalSegmentationDistanceDynamic}
                onDistanceDynamicChange={setGeneralSegmentationDistanceDynamic}
                distanceConnectivity={generalSegmentationDistanceConnectivity}
                onDistanceConnectivityChange={setGeneralSegmentationDistanceConnectivity}
                distanceSpacingZ={generalSegmentationDistanceSpacingZ}
                onDistanceSpacingZChange={setGeneralSegmentationDistanceSpacingZ}
                distanceSpacingY={generalSegmentationDistanceSpacingY}
                onDistanceSpacingYChange={setGeneralSegmentationDistanceSpacingY}
                distanceSpacingX={generalSegmentationDistanceSpacingX}
                onDistanceSpacingXChange={setGeneralSegmentationDistanceSpacingX}
                intensityProminence={generalSegmentationIntensityProminence}
                onIntensityProminenceChange={setGeneralSegmentationIntensityProminence}
                intensitySmoothingSigma={generalSegmentationIntensitySmoothingSigma}
                onIntensitySmoothingSigmaChange={setGeneralSegmentationIntensitySmoothingSigma}
                intensityLowPercentile={generalSegmentationIntensityLowPercentile}
                onIntensityLowPercentileChange={setGeneralSegmentationIntensityLowPercentile}
                intensityHighPercentile={generalSegmentationIntensityHighPercentile}
                onIntensityHighPercentileChange={setGeneralSegmentationIntensityHighPercentile}
                intensityConnectivity={generalSegmentationIntensityConnectivity}
                onIntensityConnectivityChange={setGeneralSegmentationIntensityConnectivity}
                thresholdDescription={POST_PROCESSING_PARAMETER_HELP.simpleProbabilityThreshold}
              />
            </div>
          ) : null}

          {densityControlsSelected ? (
            <>
              <div className="inferenceFormRows">
                <GpuSelectionRow
                  optionsLoading={optionsLoading}
                  availableGpus={availableGpus}
                  gpuError={gpuError}
                  selectedGpuIndex={selectedGpuIndex}
                  onSelectGpu={setSelectedGpuIndex}
                  helpDescription={POST_PROCESSING_PARAMETER_HELP.selectGpu}
                />

                <div className="postProcessingStageTitle">Stage 1: Probability density map estimation</div>

                <div className="postProcessingParameterGroup">
                  <div className="postProcessingStageDescription">
                    Use an annotated binary mask to construct Foreground/Background (FG/BG) probability distributions
                    for the spatialDINO features.
                  </div>

                  <div className="inferenceFormRow">
                    <label className="inferenceCheckboxLabel">
                      <input
                        type="checkbox"
                        checked={runDensityEstimation}
                        onChange={(event) => setRunDensityEstimation(event.target.checked)}
                      />
                      <span>Run stage 1</span>
                    </label>
                  </div>

                  {runDensityEstimation ? (
                    <>
                      {showDensityEstimationTimepoint ? (
                        <div className="inferenceFormRow">
                          <div className="inferenceFieldLabel">
                            <ParameterHelpLabel
                              label="Training timepoint"
                              description={POST_PROCESSING_PARAMETER_HELP.trainingTimepoint}
                            />
                          </div>
                          <select
                            className="inferenceSelect"
                            value={densityEstimationTimepoint}
                            onChange={(event) => setDensityEstimationTimepoint(event.target.value)}
                          >
                            {validatedSubfolderNames.map((name) => (
                              <option key={name} value={name}>
                                {name}
                              </option>
                            ))}
                          </select>
                        </div>
                      ) : null}

                      <DirectoryFieldRow
                        label={
                          <ParameterHelpLabel
                            label="Annotated FG/BG mask"
                            description={POST_PROCESSING_PARAMETER_HELP.segmentationTif}
                          />
                        }
                        path={segTifPath}
                        buttonLabel="Choose file"
                        emptyLabel="No file selected yet"
                        onChoose={() => {
                          setPickerTarget("seg_tif");
                          setPickerOpen(true);
                        }}
                      />

                      <DirectoryFieldRow
                        label={
                          <ParameterHelpLabel
                            label="Valid voxels mask (Optional)"
                            description={POST_PROCESSING_PARAMETER_HELP.validMaskTif}
                          />
                        }
                        path={validMaskTifPath}
                        buttonLabel="Choose file"
                        emptyLabel="No file selected yet"
                        onChoose={() => {
                          setPickerTarget("valid_mask");
                          setPickerOpen(true);
                        }}
                        action={
                          validMaskTifPath ? (
                            <button
                              type="button"
                              className="pickerSecondaryButton"
                              onClick={() => setValidMaskTifPath(null)}
                            >
                              Clear
                            </button>
                          ) : null
                        }
                      />
                    </>
                  ) : foregroundProbabilityMapSelected || runProbabilityMapStage2 ? (
                    <DirectoryFieldRow
                      label={
                        <ParameterHelpLabel
                          label="Stage 1 output file:"
                          description={POST_PROCESSING_PARAMETER_HELP.stage1OutputFile}
                        />
                      }
                      path={probmapDensitiesFilePath}
                      buttonLabel="Choose file"
                      emptyLabel="No file selected yet"
                      onChoose={() => {
                        setPickerTarget("probmap_densities");
                        setPickerOpen(true);
                      }}
                      action={
                        probmapDensitiesFilePath ? (
                          <button
                            type="button"
                            className="pickerSecondaryButton"
                            onClick={() => setProbmapDensitiesFilePath(null)}
                          >
                            Clear
                          </button>
                        ) : null
                      }
                    />
                  ) : null}
                </div>

                <div className="postProcessingStageTitle">
                  {foregroundProbabilityMapSelected ? "Stage 2: Generate foreground probability maps" : "Stage 2: Apply probability maps"}
                </div>

                <div className="postProcessingParameterGroup">
                  <div className="postProcessingStageDescription">
                    {foregroundProbabilityMapSelected
                      ? "Apply the densities from Stage 1 on a full movie to produce normalized foreground probability-map volumes."
                      : "Apply the probability maps from Stage 1 on a full movie to produce FG/BG masks, which are then turned into instance segmentation masks via Connected Component Labeling."}
                  </div>

                  {legacyProbabilityMapSelected ? (
                    <div className="inferenceFormRow">
                    <label className="inferenceCheckboxLabel">
                      <input
                        type="checkbox"
                        checked={runProbabilityMapStage2}
                        onChange={(event) => setRunProbabilityMapStage2(event.target.checked)}
                      />
                      <span>Run stage 2</span>
                    </label>
                    </div>
                  ) : null}
                </div>

                <div className="inferenceOptionalToggleRow">
                  <button
                    type="button"
                    className="pickerPrimaryButton"
                    aria-expanded={probabilityMapOptionalOpen}
                    aria-controls="probability-map-optional-parameters"
                    onClick={() => setProbabilityMapOptionalOpen((current) => !current)}
                  >
                    Advanced options
                  </button>
                </div>

                {probabilityMapOptionalOpen ? (
                  <div id="probability-map-optional-parameters" className="inferenceOptionalSection">
                    <div className="inferenceFormRow">
                      <div className="inferenceFieldLabel">
                        <ParameterHelpLabel
                          label="Density estimation"
                          description={POST_PROCESSING_PARAMETER_HELP.densityEstimationSettings}
                        />
                      </div>
                      <div className="inferenceInlineLabel isStrong">
                        <ParameterHelpLabel label="Method" description={POST_PROCESSING_PARAMETER_HELP.method} />
                      </div>
                      <select
                        className="inferenceSelect inferenceCompactSelect"
                        value={probabilityMapDensityMethod}
                        onChange={(event) =>
                          setProbabilityMapDensityMethod(event.target.value as "gpu-hist" | "kde")
                        }
                      >
                        <option value="gpu-hist">GPU histogram</option>
                        <option value="kde">KDE</option>
                      </select>
                      <div className="inferenceInlineLabel isStrong">
                        <ParameterHelpLabel
                          label="Density grid size"
                          description={POST_PROCESSING_PARAMETER_HELP.densityGridSize}
                        />
                      </div>
                      <PostProcessingNumberInput
                        value={probabilityMapKdePoints}
                        onChange={setProbabilityMapKdePoints}
                        min={2}
                        step={1}
                      />
                      {probabilityMapDensityMethod === "gpu-hist" ? (
                        <>
                          <div className="inferenceInlineLabel isStrong">
                            <ParameterHelpLabel
                              label="Histogram sigma"
                              description={POST_PROCESSING_PARAMETER_HELP.histogramSigma}
                            />
                          </div>
                          <PostProcessingNumberInput
                            value={probabilityMapHistSigmaBins}
                            onChange={setProbabilityMapHistSigmaBins}
                            min={0.1}
                            step={0.1}
                          />
                        </>
                      ) : (
                        <>
                          <div className="inferenceInlineLabel isStrong">
                            <ParameterHelpLabel
                              label="KDE bandwidth"
                              description={POST_PROCESSING_PARAMETER_HELP.kdeBandwidth}
                            />
                          </div>
                          <PostProcessingTextInput
                            value={probabilityMapKdeBandwidth}
                            onChange={setProbabilityMapKdeBandwidth}
                            placeholder="auto"
                          />
                        </>
                      )}
                    </div>

                    <div className="inferenceFormRow">
                      <div className="inferenceFieldLabel">
                        <ParameterHelpLabel label="Sampling" description={POST_PROCESSING_PARAMETER_HELP.sampling} />
                      </div>
                      <div className="inferenceInlineLabel isStrong">
                        <ParameterHelpLabel
                          label="Max samples per class"
                          description={POST_PROCESSING_PARAMETER_HELP.maxSamplesPerClass}
                        />
                      </div>
                      <PostProcessingNumberInput
                        value={probabilityMapKdeMaxSamples}
                        onChange={setProbabilityMapKdeMaxSamples}
                        min={1}
                        step={1}
                      />
                      <div className="inferenceInlineLabel isStrong">
                        <ParameterHelpLabel label="Random seed" description={POST_PROCESSING_PARAMETER_HELP.randomSeed} />
                      </div>
                      <PostProcessingNumberInput
                        value={probabilityMapSeed}
                        onChange={setProbabilityMapSeed}
                        min={0}
                        step={1}
                      />
                    </div>

                    <div className="inferenceFormRow">
                      <div className="inferenceFieldLabel">
                        <ParameterHelpLabel
                          label="Probability map estimation"
                          description={POST_PROCESSING_PARAMETER_HELP.probabilityMapEstimation}
                        />
                      </div>
                      <div className="inferenceInlineLabel isStrong">
                        <ParameterHelpLabel
                          label="Feature batch"
                          description={POST_PROCESSING_PARAMETER_HELP.featureBatch}
                        />
                      </div>
                      <PostProcessingNumberInput
                        value={probabilityMapFeatureBatch}
                        onChange={setProbabilityMapFeatureBatch}
                        min={1}
                        step={1}
                      />
                      {legacyProbabilityMapSelected ? (
                        <>
                          <div className="inferenceInlineLabel isStrong">
                            <ParameterHelpLabel
                              label="BG probability threshold"
                              description={POST_PROCESSING_PARAMETER_HELP.bgProbabilityThreshold}
                            />
                          </div>
                          <PostProcessingNumberInput
                            value={probabilityMapBgThreshold}
                            onChange={setProbabilityMapBgThreshold}
                            min={0}
                            step={0.01}
                          />
                          <div className="inferenceInlineLabel isStrong">
                            <ParameterHelpLabel
                              label="FG probability threshold"
                              description={POST_PROCESSING_PARAMETER_HELP.fgProbabilityThreshold}
                            />
                          </div>
                          <PostProcessingNumberInput
                            value={probabilityMapFgThreshold}
                            onChange={setProbabilityMapFgThreshold}
                            min={0}
                            step={0.01}
                          />
                        </>
                      ) : null}
                    </div>
                  </div>
                ) : null}
              </div>
            </>
          ) : null}

          <div className="validationActions">
            <button
              type="button"
              className="preprocessValidateButton"
              onClick={() =>
                foregroundProbabilityMapSelected ? void handleForegroundProbabilityMapRun() : void handleSegmentationRun()
              }
              disabled={submitting}
            >
              {submitting ? "Submitting..." : "Run"}
            </button>
          </div>

          {runFeedback ? <ValidationMessage tone={runFeedback.tone}>{runFeedback.message}</ValidationMessage> : null}
        </section>
      ) : null}

      {parametersVisible && trackingSelected ? (
        <section className="datasetCard inferenceFormCard" aria-label="Tracking parameters">
          <div className="sidebarHint">
            Tracking reads <code>&lt;inference output folder&gt;/lr_feats/&lt;timepoint&gt;.npy</code> and{" "}
            <code>&lt;inference output folder&gt;/raw/&lt;timepoint&gt;.tif</code>, matches them against{" "}
            <code>&lt;segmentation output folder&gt;/&lt;timepoint&gt;.tif</code>, follows the original
            overlap-and-voting tracker, and writes a single final-format CSV into the selected
            output folder while keeping exported <code>z</code> coordinates as-is.
          </div>

          {trackingOptionalOpen ? (
            <div id="tracking-optional-parameters" className="inferenceOptionalSection">
              <div className="inferenceFormRows">
                <div className="inferenceFormRow">
                  <div className="inferenceFieldLabel">
                    <ParameterHelpLabel
                      label="Save extended results"
                      description={POST_PROCESSING_PARAMETER_HELP.saveExtendedResults}
                    />
                  </div>
                  <label className="inferenceCheckboxLabel">
                    <input
                      type="checkbox"
                      checked={trackingSaveExtendedResults}
                      onChange={(event) => setTrackingSaveExtendedResults(event.target.checked)}
                    />
                    <span>Enabled</span>
                  </label>
                </div>

                <div className="inferenceFormRow">
                  <div className="inferenceFieldLabel">
                    <ParameterHelpLabel label="Ignore features" description={POST_PROCESSING_PARAMETER_HELP.ignoreFeatures} />
                  </div>
                  <label className="inferenceCheckboxLabel">
                    <input
                      type="checkbox"
                      checked={trackingIgnoreFeatures}
                      onChange={(event) => setTrackingIgnoreFeatures(event.target.checked)}
                    />
                    <span>Enabled</span>
                  </label>
                </div>

                <div className="inferenceFormRow">
                  <div className="inferenceFieldLabel">
                    <ParameterHelpLabel
                      label="Disable centroid fallback"
                      description={POST_PROCESSING_PARAMETER_HELP.disableCentroidFallback}
                    />
                  </div>
                  <label className="inferenceCheckboxLabel">
                    <input
                      type="checkbox"
                      checked={trackingDisableCentroidFallback}
                      onChange={(event) => setTrackingDisableCentroidFallback(event.target.checked)}
                    />
                    <span>Enabled</span>
                  </label>
                </div>

                <div className="inferenceFormRow">
                  <div className="inferenceFieldLabel">
                    <ParameterHelpLabel
                      label="Aggressive feature matching"
                      description={POST_PROCESSING_PARAMETER_HELP.aggressiveFeatureMatching}
                    />
                  </div>
                  <label className="inferenceCheckboxLabel">
                    <input
                      type="checkbox"
                      checked={trackingAggressiveFeatureMatching}
                      onChange={(event) => setTrackingAggressiveFeatureMatching(event.target.checked)}
                    />
                    <span>Enabled</span>
                  </label>
                  <div className="inferenceInlineLabel isStrong">
                    <ParameterHelpLabel label="Min feature votes" description={POST_PROCESSING_PARAMETER_HELP.minFeatureVotes} />
                  </div>
                  <PostProcessingNumberInput
                    value={trackingMinFeatureVotes}
                    onChange={setTrackingMinFeatureVotes}
                    min={1}
                    step={1}
                  />
                </div>

                <div className="inferenceFormRow">
                  <div className="inferenceFieldLabel">
                    <ParameterHelpLabel label="Match window" description={POST_PROCESSING_PARAMETER_HELP.matchWindow} />
                  </div>
                  <div className="inferenceInlineLabel isStrong">
                    <ParameterHelpLabel label="Max XY distance" description={POST_PROCESSING_PARAMETER_HELP.maxXyDistance} />
                  </div>
                  <PostProcessingNumberInput
                    value={trackingMaxDistanceXy}
                    onChange={setTrackingMaxDistanceXy}
                    min={0.1}
                    step={0.1}
                  />
                  <div className="inferenceInlineLabel">voxels</div>
                  <div className="inferenceInlineLabel isStrong">
                    <ParameterHelpLabel label="Max Z distance" description={POST_PROCESSING_PARAMETER_HELP.maxZDistance} />
                  </div>
                  <PostProcessingNumberInput
                    value={trackingMaxDistanceZ}
                    onChange={setTrackingMaxDistanceZ}
                    min={0.1}
                    step={0.1}
                  />
                  <div className="inferenceInlineLabel">voxels</div>
                </div>

                <div className="inferenceFormRow">
                  <div className="inferenceFieldLabel">
                    <ParameterHelpLabel
                      label="Distance logic"
                      description={POST_PROCESSING_PARAMETER_HELP.distanceLogic}
                    />
                  </div>
                  <div className="inferenceInlineLabel isStrong">
                    <ParameterHelpLabel
                      label="Z distance weight"
                      description={POST_PROCESSING_PARAMETER_HELP.zDistanceWeight}
                    />
                  </div>
                  <PostProcessingNumberInput
                    value={trackingZDistanceWeight}
                    onChange={setTrackingZDistanceWeight}
                    min={0.1}
                    step={0.1}
                  />
                  <div className="inferenceInlineLabel isStrong">
                    <ParameterHelpLabel
                      label="Immediate assignment distance"
                      description={POST_PROCESSING_PARAMETER_HELP.immediateAssignmentDistance}
                    />
                  </div>
                  <PostProcessingNumberInput
                    value={trackingMinDistanceToRemoveCand}
                    onChange={setTrackingMinDistanceToRemoveCand}
                    min={0}
                    step={0.1}
                  />
                </div>

                <div className="inferenceFormRow">
                  <div className="inferenceFieldLabel">
                    <ParameterHelpLabel
                      label="Voting thresholds"
                      description={POST_PROCESSING_PARAMETER_HELP.votingThresholds}
                    />
                  </div>
                  <div className="inferenceInlineLabel isStrong">
                    <ParameterHelpLabel label="Vote thresholds" description={POST_PROCESSING_PARAMETER_HELP.voteThresholds} />
                  </div>
                  <PostProcessingTextInput value={trackingVoteThresholds} onChange={setTrackingVoteThresholds} />
                  <div className="inferenceInlineLabel isStrong">
                    <ParameterHelpLabel label="Min Dice" description={POST_PROCESSING_PARAMETER_HELP.minDice} />
                  </div>
                  <PostProcessingNumberInput
                    value={trackingDiceThreshold}
                    onChange={setTrackingDiceThreshold}
                    min={0}
                    step={0.01}
                  />
                  <div className="inferenceInlineLabel isStrong">
                    <ParameterHelpLabel label="Min Corr" description={POST_PROCESSING_PARAMETER_HELP.minCorr} />
                  </div>
                  <PostProcessingNumberInput
                    value={trackingCorrThreshold}
                    onChange={setTrackingCorrThreshold}
                    min={-1}
                    step={0.01}
                  />
                </div>
              </div>
            </div>
          ) : null}

          <div className="validationActions">
            <button
              type="button"
              className="preprocessValidateButton"
              onClick={() => void handleTrackingRun()}
              disabled={submitting}
            >
              {submitting ? "Submitting..." : "Run"}
            </button>
          </div>

          {runFeedback ? <ValidationMessage tone={runFeedback.tone}>{runFeedback.message}</ValidationMessage> : null}
        </section>
      ) : null}

      <ServerDirectoryPicker
        open={pickerOpen}
        title={pickerTitle}
        selectionMode={
          pickerTarget === "seg_tif" || pickerTarget === "valid_mask" || pickerTarget === "probmap_densities"
            ? "file"
            : "directory"
        }
        initialPath={pickerInitialPath}
        onClose={() => setPickerOpen(false)}
        onSelect={(path) => {
          if (pickerTarget === "input") {
            setInputPath(path);
          } else if (pickerTarget === "process_output") {
            setProcessFeaturesOutputPath(path);
          } else if (pickerTarget === "segmentation_output") {
            setSegmentationOutputPath(path);
          } else if (pickerTarget === "tracking_segmentation") {
            setTrackingSegmentationPath(path);
          } else if (pickerTarget === "tracking_output") {
            setTrackingOutputPath(path);
          } else if (pickerTarget === "seg_tif") {
            setSegTifPath(path);
          } else if (pickerTarget === "probmap_densities") {
            setProbmapDensitiesFilePath(path);
          } else {
            setValidMaskTifPath(path);
          }
          setPickerOpen(false);
        }}
      />
    </div>
  );
}

function GeneralSegmentationPreview({
  inputPath,
  gpuIndex,
  enabled,
  sources,
  source,
  sourceId,
  onSourceIdChange,
  thresholdComponent,
  onThresholdComponentChange,
  displaySource,
  onDisplaySourceChange,
  processData,
  onProcessDataChange,
  dataOperations,
  onDataOperationsChange,
  threshold,
  onThresholdChange,
  invertMask,
  onInvertMaskChange,
  processMask,
  onProcessMaskChange,
  maskOperations,
  onMaskOperationsChange,
  instanceMethod,
  onInstanceMethodChange,
  voronoiSpotSigma,
  onVoronoiSpotSigmaChange,
  voronoiOutlineSigma,
  onVoronoiOutlineSigmaChange,
  distanceDynamic,
  onDistanceDynamicChange,
  distanceConnectivity,
  onDistanceConnectivityChange,
  distanceSpacingZ,
  onDistanceSpacingZChange,
  distanceSpacingY,
  onDistanceSpacingYChange,
  distanceSpacingX,
  onDistanceSpacingXChange,
  intensityProminence,
  onIntensityProminenceChange,
  intensitySmoothingSigma,
  onIntensitySmoothingSigmaChange,
  intensityLowPercentile,
  onIntensityLowPercentileChange,
  intensityHighPercentile,
  onIntensityHighPercentileChange,
  intensityConnectivity,
  onIntensityConnectivityChange,
  thresholdDescription,
}: {
  inputPath: string | null;
  gpuIndex: number | null;
  enabled: boolean;
  sources: GeneralSegmentationSource[];
  source: GeneralSegmentationSource | null;
  sourceId: string;
  onSourceIdChange: (value: string) => void;
  thresholdComponent: number;
  onThresholdComponentChange: (value: number) => void;
  displaySource: GeneralSegmentationDisplaySource;
  onDisplaySourceChange: (value: GeneralSegmentationDisplaySource) => void;
  processData: boolean;
  onProcessDataChange: (value: boolean) => void;
  dataOperations: GeneralSegmentationDataOperation[];
  onDataOperationsChange: (value: GeneralSegmentationDataOperation[]) => void;
  threshold: string;
  onThresholdChange: (value: string) => void;
  invertMask: boolean;
  onInvertMaskChange: (value: boolean) => void;
  processMask: boolean;
  onProcessMaskChange: (value: boolean) => void;
  maskOperations: GeneralSegmentationMaskOperation[];
  onMaskOperationsChange: (value: GeneralSegmentationMaskOperation[]) => void;
  instanceMethod: GeneralSegmentationInstanceMethod;
  onInstanceMethodChange: (value: GeneralSegmentationInstanceMethod) => void;
  voronoiSpotSigma: string;
  onVoronoiSpotSigmaChange: (value: string) => void;
  voronoiOutlineSigma: string;
  onVoronoiOutlineSigmaChange: (value: string) => void;
  distanceDynamic: string;
  onDistanceDynamicChange: (value: string) => void;
  distanceConnectivity: 6 | 26;
  onDistanceConnectivityChange: (value: 6 | 26) => void;
  distanceSpacingZ: string;
  onDistanceSpacingZChange: (value: string) => void;
  distanceSpacingY: string;
  onDistanceSpacingYChange: (value: string) => void;
  distanceSpacingX: string;
  onDistanceSpacingXChange: (value: string) => void;
  intensityProminence: string;
  onIntensityProminenceChange: (value: string) => void;
  intensitySmoothingSigma: string;
  onIntensitySmoothingSigmaChange: (value: string) => void;
  intensityLowPercentile: string;
  onIntensityLowPercentileChange: (value: string) => void;
  intensityHighPercentile: string;
  onIntensityHighPercentileChange: (value: string) => void;
  intensityConnectivity: 6 | 26;
  onIntensityConnectivityChange: (value: 6 | 26) => void;
  thresholdDescription: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const lastDefaultKeyRef = useRef("");
  const [metadata, setMetadata] = useState<GeneralSegmentationPreviewMetadataResponse | null>(null);
  const [metadataLoading, setMetadataLoading] = useState(false);
  const [metadataError, setMetadataError] = useState<string | null>(null);
  const [selectedTimepoint, setSelectedTimepoint] = useState("");
  const [selectedView, setSelectedView] = useState<GeneralSegmentationPreviewView>("slice");
  const [displayContrast, setDisplayContrast] = useState<GeneralSegmentationDisplayContrast>("auto");
  const [zIndex, setZIndex] = useState(0);
  const [maskVisible, setMaskVisible] = useState(true);
  const [frame, setFrame] = useState<GeneralSegmentationPreviewFrame | null>(null);
  const [frameLoading, setFrameLoading] = useState(false);
  const [frameError, setFrameError] = useState<string | null>(null);
  const [thresholdRange, setThresholdRange] = useState<GeneralSegmentationThresholdRange | null>(null);
  const [rangeLoading, setRangeLoading] = useState(false);
  const [rangeError, setRangeError] = useState<string | null>(null);
  const [computedDataKey, setComputedDataKey] = useState<string | null>(null);
  const [dataComputeLoading, setDataComputeLoading] = useState(false);
  const [dataComputeError, setDataComputeError] = useState<string | null>(null);
  const [binaryMaskFrame, setBinaryMaskFrame] = useState<GeneralSegmentationPreviewMaskFrame | null>(null);
  const [maskError, setMaskError] = useState<string | null>(null);
  const [processedMaskFrame, setProcessedMaskFrame] = useState<GeneralSegmentationPreviewMaskFrame | null>(null);
  const [maskComputeLoading, setMaskComputeLoading] = useState(false);
  const [maskComputeError, setMaskComputeError] = useState<string | null>(null);
  const [instanceMaskFrame, setInstanceMaskFrame] = useState<GeneralSegmentationPreviewMaskFrame | null>(null);
  const [instanceLoading, setInstanceLoading] = useState(false);
  const [instanceError, setInstanceError] = useState<string | null>(null);
  const [foregroundPixels, setForegroundPixels] = useState(0);

  const timepoints = metadata?.timepoints ?? [];
  const compatibleTimepoints = timepoints.filter((timepoint) => timepoint.compatible);
  const selectedTimepointMetadata =
    compatibleTimepoints.find((timepoint) => timepoint.name === selectedTimepoint) ?? null;
  const zMax = Math.max(0, (selectedTimepointMetadata?.zCount ?? 1) - 1);
  const maxComponent = Math.max(0, (source?.componentCount ?? 1) - 1);
  const clampedThresholdComponent = clampInteger(thresholdComponent, 0, maxComponent);
  const hasSourceComponents = source !== null && source.kind !== "raw" && source.componentCount > 1;
  const sourceComponentLabel = (index: number) =>
    source?.componentNames?.[index] ?? (source?.kind === "pca" ? `PCA #${index + 1}` : `Component #${index + 1}`);
  const previewDataOperations = useMemo(
    () => serializeGeneralDataOperations(processData, dataOperations, false).operations,
    [dataOperations, processData]
  );
  const previewMaskOperations = useMemo(
    () => serializeGeneralMaskOperations(processMask, maskOperations, false).operations,
    [maskOperations, processMask]
  );
  const previewDataOperationsKey = JSON.stringify(previewDataOperations);
  const previewMaskOperationsKey = JSON.stringify(previewMaskOperations);
  const previewVoronoiSpotSigma = parseNonnegativeNumber(voronoiSpotSigma) ?? 2;
  const previewVoronoiOutlineSigma = parseNonnegativeNumber(voronoiOutlineSigma) ?? 2;
  const previewDistanceDynamic = parseNonnegativeNumber(distanceDynamic) ?? 1;
  const previewDistanceSpacingZ = parsePositiveNumber(distanceSpacingZ) ?? 1;
  const previewDistanceSpacingY = parsePositiveNumber(distanceSpacingY) ?? 1;
  const previewDistanceSpacingX = parsePositiveNumber(distanceSpacingX) ?? 1;
  const parsedIntensityProminence = parseNonnegativeNumber(intensityProminence);
  const previewIntensityProminence =
    parsedIntensityProminence !== null && parsedIntensityProminence <= 1 ? parsedIntensityProminence : 0.15;
  const previewIntensitySmoothingSigma = parseNonnegativeNumber(intensitySmoothingSigma) ?? 0;
  const parsedIntensityLowPercentile = parseNonnegativeNumber(intensityLowPercentile);
  const parsedIntensityHighPercentile = parseNonnegativeNumber(intensityHighPercentile);
  const validIntensityPercentiles =
    parsedIntensityLowPercentile !== null &&
    parsedIntensityHighPercentile !== null &&
    parsedIntensityLowPercentile <= 100 &&
    parsedIntensityHighPercentile <= 100 &&
    parsedIntensityLowPercentile < parsedIntensityHighPercentile;
  const previewIntensityLowPercentile = validIntensityPercentiles ? parsedIntensityLowPercentile : 1;
  const previewIntensityHighPercentile = validIntensityPercentiles ? parsedIntensityHighPercentile : 99;
  const currentZIndex = selectedView === "slice" ? clampInteger(zIndex, 0, zMax) : null;
  const requestedDataProcessorKey = previewDataOperations.length > 0 && gpuIndex !== null ? `gpu:${gpuIndex}` : "cpu";
  const dataBaseKey = JSON.stringify({
    inputPath,
    sourceId: source?.id ?? sourceId,
    component: clampedThresholdComponent,
    timepoint: selectedTimepoint,
  });
  const requestedDataKey = JSON.stringify({
    dataBaseKey,
    dataOperations: previewDataOperations,
    processor: requestedDataProcessorKey,
  });
  const dataProcessingComputed = processData && computedDataKey === requestedDataKey;
  const activeDataOperations = dataProcessingComputed ? previewDataOperations : [];
  const activeDataProcessorKey = activeDataOperations.length > 0 && gpuIndex !== null ? `gpu:${gpuIndex}` : "cpu";
  const activeDataOperationsKey = JSON.stringify(activeDataOperations);
  const activeDataKey = JSON.stringify({
    dataBaseKey,
    dataOperations: activeDataOperations,
    processor: activeDataProcessorKey,
  });
  const activeEvaluationKey = JSON.stringify({
    activeDataKey,
    view: selectedView,
    zIndex: currentZIndex,
  });
  const activeSurfaceKey = JSON.stringify({
    activeEvaluationKey,
    displaySource,
    displayContrast,
  });
  const thresholdMin = thresholdRange?.key === activeDataKey ? thresholdRange.min : source?.thresholdMin ?? 0;
  const thresholdMax = thresholdRange?.key === activeDataKey ? thresholdRange.max : source?.thresholdMax ?? 1;
  const thresholdStep = thresholdRange?.key === activeDataKey ? thresholdRange.step : source?.thresholdStep ?? 0.01;
  const parsedThreshold = Number.parseFloat(threshold);
  const thresholdValue = Number.isFinite(parsedThreshold) ? parsedThreshold : (Number(thresholdMin) + Number(thresholdMax)) / 2;
  const effectiveThresholdValue = clampNumber(thresholdValue, Number(thresholdMin), Number(thresholdMax));
  const sliderThresholdValue = effectiveThresholdValue;
  const binaryMaskKey = JSON.stringify({
    evaluationKey: activeEvaluationKey,
    threshold: effectiveThresholdValue,
    invertMask,
  });
  const processedMaskKey = JSON.stringify({
    binaryMaskKey,
    maskOperations: previewMaskOperations,
  });
  const instanceMaskKey = JSON.stringify({
    processedMaskKey,
    instanceMethod,
    voronoiSpotSigma: previewVoronoiSpotSigma,
    voronoiOutlineSigma: previewVoronoiOutlineSigma,
    distanceDynamic: previewDistanceDynamic,
    distanceConnectivity,
    distanceSpacingZ: previewDistanceSpacingZ,
    distanceSpacingY: previewDistanceSpacingY,
    distanceSpacingX: previewDistanceSpacingX,
    intensityProminence: previewIntensityProminence,
    intensitySmoothingSigma: previewIntensitySmoothingSigma,
    intensityLowPercentile: previewIntensityLowPercentile,
    intensityHighPercentile: previewIntensityHighPercentile,
    intensityConnectivity,
  });
  const validBinaryMaskFrame = binaryMaskFrame?.key === binaryMaskKey ? binaryMaskFrame : null;
  const fallbackBinaryMaskFrame =
    binaryMaskFrame?.evaluationKey === activeEvaluationKey &&
    binaryMaskFrame.width === frame?.width &&
    binaryMaskFrame.height === frame?.height
      ? binaryMaskFrame
      : null;
  const activeMaskFrame =
    instanceMethod !== "none" && instanceMaskFrame?.key === instanceMaskKey
      ? instanceMaskFrame
      : processMask && processedMaskFrame?.key === processedMaskKey
        ? processedMaskFrame
      : validBinaryMaskFrame ?? fallbackBinaryMaskFrame;
  const totalPixels = activeMaskFrame ? activeMaskFrame.width * activeMaskFrame.height : frame ? frame.width * frame.height : 0;
  const foregroundPercent = totalPixels > 0 ? (foregroundPixels / totalPixels) * 100 : 0;
  const instancePreviewState =
    instanceMethod === "none"
      ? null
      : instanceLoading
        ? "Computing..."
        : instanceMaskFrame?.key === instanceMaskKey
          ? "Computed"
          : "Uncomputed";
  const dataProcessState = !processData
    ? null
    : dataComputeLoading
      ? "Computing..."
      : dataProcessingComputed
        ? "Computed"
        : "Uncomputed";
  const maskProcessState = !processMask
    ? null
    : maskComputeLoading
      ? "Computing..."
      : processedMaskFrame?.key === processedMaskKey
        ? "Computed"
        : "Uncomputed";

  useEffect(() => {
    if (!enabled || !inputPath) {
      setMetadata(null);
      setMetadataLoading(false);
      setMetadataError(null);
      setSelectedTimepoint("");
      setFrame(null);
      setThresholdRange(null);
      setComputedDataKey(null);
      setBinaryMaskFrame(null);
      setProcessedMaskFrame(null);
      setInstanceMaskFrame(null);
      setRangeError(null);
      setDataComputeError(null);
      setMaskError(null);
      setMaskComputeError(null);
      setInstanceError(null);
      lastDefaultKeyRef.current = "";
      return;
    }

    const controller = new AbortController();
    setMetadataLoading(true);
    setMetadataError(null);
    setMetadata(null);
    setFrame(null);
    setThresholdRange(null);
    setComputedDataKey(null);
    setBinaryMaskFrame(null);
    setProcessedMaskFrame(null);
    setInstanceMaskFrame(null);
    setRangeError(null);
    setDataComputeError(null);
    setMaskError(null);
    setMaskComputeError(null);
    setInstanceError(null);

    async function loadMetadata() {
      try {
        const resp = await fetch("/api/post-processing/general-segmentation/preview/metadata", {
          method: "POST",
          headers: {
            Accept: "application/json",
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ input_path: inputPath }),
          signal: controller.signal,
        });
        if (!resp.ok) {
          const detail = await safeJson(resp);
          const message =
            detail && typeof detail === "object" && "detail" in detail && typeof detail.detail === "string"
              ? detail.detail
              : `Preview metadata failed: ${resp.status} ${resp.statusText}`;
          throw new Error(message);
        }

        const json = (await resp.json()) as GeneralSegmentationPreviewMetadataResponse;
        if (!json.valid) {
          throw new Error(json.message || "Preview metadata is not available.");
        }
        setMetadata(json);
      } catch (error) {
        if (controller.signal.aborted) return;
        setMetadataError(error instanceof Error ? error.message : "Unknown preview metadata error");
      } finally {
        if (!controller.signal.aborted) {
          setMetadataLoading(false);
        }
      }
    }

    void loadMetadata();
    return () => controller.abort();
  }, [enabled, inputPath]);

  useEffect(() => {
    if (!metadata) return;
    const compatible = metadata.timepoints?.filter((timepoint) => timepoint.compatible) ?? [];
    if (!compatible.length) return;
    if (selectedTimepoint && compatible.some((timepoint) => timepoint.name === selectedTimepoint)) return;

    const defaultTimepoint =
      metadata.defaultTimepoint && compatible.some((timepoint) => timepoint.name === metadata.defaultTimepoint)
        ? metadata.defaultTimepoint
        : compatible[0].name;
    setSelectedTimepoint(defaultTimepoint);
  }, [metadata, selectedTimepoint]);

  useEffect(() => {
    if (!selectedTimepointMetadata?.zCount) return;
    setZIndex((current) => clampInteger(current, 0, selectedTimepointMetadata.zCount! - 1));
  }, [selectedTimepointMetadata?.zCount]);

  useEffect(() => {
    if (!enabled || !inputPath || !source || !selectedTimepoint || !selectedTimepointMetadata) {
      setFrame(null);
      setFrameLoading(false);
      setFrameError(null);
      return;
    }

    const controller = new AbortController();
    const selectedSource = source;
    const requestSurfaceKey = activeSurfaceKey;
    const requestEvaluationKey = activeEvaluationKey;
    setFrameLoading(true);
    setFrameError(null);

    async function loadFrame() {
      try {
        const requestBody = {
          input_path: inputPath,
          gpu_index: gpuIndex,
          source_id: selectedSource.id,
          threshold_component_index: clampedThresholdComponent,
          display_source: displaySource,
          display_contrast: displayContrast,
          data_operations: activeDataOperations,
          require_cached_data: activeDataOperations.length > 0,
          timepoint: selectedTimepoint,
          view: selectedView,
          z_index: currentZIndex,
        };
        const resp = await fetch("/api/post-processing/general-segmentation/preview/surface", {
          method: "POST",
          headers: {
            Accept: "application/json",
            "Content-Type": "application/json",
          },
          body: JSON.stringify(requestBody),
          signal: controller.signal,
        });
        if (!resp.ok) {
          const detail = await safeJson(resp);
          const message =
            detail && typeof detail === "object" && "detail" in detail && typeof detail.detail === "string"
              ? detail.detail
              : `Preview surface failed: ${resp.status} ${resp.statusText}`;
          throw new Error(message);
        }

        const json = (await resp.json()) as GeneralSegmentationPreviewSurfaceResponse;
        const display = decodeBase64Bytes(json.display.data);
        const pixelCount = json.width * json.height;
        if (display.length !== pixelCount) {
          throw new Error("Preview payload size does not match its dimensions.");
        }
        const evaluationValues =
          json.evaluation.kind === "slice" ? decodeFloat32Base64(json.evaluation.values.data) : null;
        const evaluationMaxValues =
          json.evaluation.kind === "projection" ? decodeFloat32Base64(json.evaluation.maxValues.data) : null;
        const evaluationMinValues =
          json.evaluation.kind === "projection" ? decodeFloat32Base64(json.evaluation.minValues.data) : null;
        if (
          (evaluationValues !== null && evaluationValues.length !== pixelCount) ||
          (evaluationMaxValues !== null && evaluationMaxValues.length !== pixelCount) ||
          (evaluationMinValues !== null && evaluationMinValues.length !== pixelCount)
        ) {
          throw new Error("Preview evaluation payload size does not match its dimensions.");
        }
        setFrame({
          key: requestSurfaceKey,
          evaluationKey: requestEvaluationKey,
          timepoint: json.timepoint,
          sourceId: json.sourceId,
          view: json.view,
          zIndex: json.zIndex,
          width: json.width,
          height: json.height,
          shape: json.shape,
          display,
          evaluationKind: json.evaluation.kind,
          evaluationValues,
          evaluationMaxValues,
          evaluationMinValues,
          displayLow: json.display.displayLow,
          displayHigh: json.display.displayHigh,
        });
      } catch (error) {
        if (controller.signal.aborted) return;
        setFrame(null);
        setFrameError(error instanceof Error ? error.message : "Unknown preview image error");
      } finally {
        if (!controller.signal.aborted) {
          setFrameLoading(false);
        }
      }
    }

    void loadFrame();
    return () => controller.abort();
  }, [
    enabled,
    inputPath,
    source,
    selectedTimepoint,
    selectedTimepointMetadata,
    selectedView,
    currentZIndex,
    clampedThresholdComponent,
    displaySource,
    displayContrast,
    gpuIndex,
    activeSurfaceKey,
    activeEvaluationKey,
    activeDataOperationsKey,
  ]);

  useEffect(() => {
    if (!frame) {
      setMaskError(null);
      return;
    }
    if (frame.evaluationKey !== activeEvaluationKey) {
      setMaskError(null);
      return;
    }

    const pixelCount = frame.width * frame.height;
    const mask = new Uint32Array(pixelCount);
    setMaskError(null);

    if (frame.evaluationKind === "slice") {
      const values = frame.evaluationValues;
      if (values === null || values.length !== pixelCount) {
        setBinaryMaskFrame(null);
        setMaskError("Preview evaluation plane is not available.");
        return;
      }
      for (let index = 0; index < pixelCount; index += 1) {
        const value = values[index];
        const foreground = Number.isFinite(value) && (invertMask ? value < effectiveThresholdValue : value >= effectiveThresholdValue);
        mask[index] = foreground ? 1 : 0;
      }
    } else {
      const values = invertMask ? frame.evaluationMinValues : frame.evaluationMaxValues;
      if (values === null || values.length !== pixelCount) {
        setBinaryMaskFrame(null);
        setMaskError("Preview evaluation projection is not available.");
        return;
      }
      for (let index = 0; index < pixelCount; index += 1) {
        const value = values[index];
        const foreground = Number.isFinite(value) && (invertMask ? value < effectiveThresholdValue : value >= effectiveThresholdValue);
        mask[index] = foreground ? 1 : 0;
      }
    }

    setBinaryMaskFrame({
      key: binaryMaskKey,
      evaluationKey: frame.evaluationKey,
      timepoint: frame.timepoint,
      sourceId: frame.sourceId,
      view: frame.view,
      zIndex: frame.zIndex,
      width: frame.width,
      height: frame.height,
      shape: frame.shape,
      mask,
      maskIsInstance: false,
    });
  }, [
    binaryMaskKey,
    activeEvaluationKey,
    frame,
    effectiveThresholdValue,
    invertMask,
  ]);

  useEffect(() => {
    if (!enabled || !inputPath || !source || !selectedTimepoint || !selectedTimepointMetadata) {
      setThresholdRange(null);
      setRangeLoading(false);
      setRangeError(null);
      return;
    }

    const controller = new AbortController();
    const requestKey = activeDataKey;
    const selectedSource = source;
    setRangeLoading(true);
    setRangeError(null);

    async function loadRange() {
      try {
        const requestBody = {
          input_path: inputPath,
          gpu_index: gpuIndex,
          source_id: selectedSource.id,
          threshold_component_index: clampedThresholdComponent,
          data_operations: activeDataOperations,
          require_cached_data: activeDataOperations.length > 0,
          timepoint: selectedTimepoint,
        };
        const resp = await fetch("/api/post-processing/general-segmentation/preview/data", {
          method: "POST",
          headers: {
            Accept: "application/json",
            "Content-Type": "application/json",
          },
          body: JSON.stringify(requestBody),
          signal: controller.signal,
        });
        if (!resp.ok) {
          const detail = await safeJson(resp);
          const message =
            detail && typeof detail === "object" && "detail" in detail && typeof detail.detail === "string"
              ? detail.detail
              : `Preview data failed: ${resp.status} ${resp.statusText}`;
          throw new Error(message);
        }

        const json = (await resp.json()) as GeneralSegmentationPreviewDataResponse;
        setThresholdRange({
          key: requestKey,
          min: json.rangeMin,
          max: json.rangeMax,
          step: json.step,
        });
      } catch (error) {
        if (controller.signal.aborted) return;
        if (activeDataOperations.length > 0) {
          setComputedDataKey(null);
        }
        setRangeError(error instanceof Error ? error.message : "Unknown preview data error");
      } finally {
        if (!controller.signal.aborted) {
          setRangeLoading(false);
        }
      }
    }

    void loadRange();
    return () => controller.abort();
  }, [
    activeDataKey,
    activeDataOperationsKey,
    clampedThresholdComponent,
    enabled,
    gpuIndex,
    inputPath,
    selectedTimepoint,
    selectedTimepointMetadata,
    source,
  ]);

  useEffect(() => {
    if (!source || !inputPath || thresholdRange?.key !== activeDataKey) return;
    const defaultKey = activeDataKey;
    if (lastDefaultKeyRef.current === defaultKey) return;
    lastDefaultKeyRef.current = defaultKey;
    const midpoint = (thresholdMin + thresholdMax) / 2;
    const defaultValue =
      activeDataOperations.length > 0 ? midpoint : source.kind === "probmap" ? 0.5 : source.kind === "pca" ? 128 : midpoint;
    onThresholdChange(formatThresholdValue(defaultValue, thresholdStep));
  }, [
    activeDataOperations.length,
    clampedThresholdComponent,
    activeDataKey,
    inputPath,
    onThresholdChange,
    source,
    thresholdRange,
    thresholdMax,
    thresholdMin,
    thresholdStep,
  ]);

  useEffect(() => {
    if (!enabled || thresholdRange?.key !== activeDataKey || !Number.isFinite(parsedThreshold)) return;
    if (parsedThreshold === effectiveThresholdValue) return;
    onThresholdChange(formatThresholdValue(effectiveThresholdValue, thresholdStep));
  }, [
    activeDataKey,
    effectiveThresholdValue,
    enabled,
    onThresholdChange,
    parsedThreshold,
    thresholdRange,
    thresholdStep,
  ]);

  useEffect(() => {
    if (processMask) return;
    setProcessedMaskFrame(null);
    setMaskComputeError(null);
    setMaskComputeLoading(false);
  }, [processMask]);

  useEffect(() => {
    if (instanceMethod !== "none") return;
    setInstanceMaskFrame(null);
    setInstanceError(null);
    setInstanceLoading(false);
  }, [instanceMethod]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !frame) return;

    canvas.width = frame.width;
    canvas.height = frame.height;
    const context = canvas.getContext("2d");
    if (!context) return;

    const imageData = context.createImageData(frame.width, frame.height);
    const output = imageData.data;
    let currentForegroundPixels = 0;
    const mask = activeMaskFrame?.width === frame.width && activeMaskFrame.height === frame.height ? activeMaskFrame.mask : null;
    const maskIsInstance = Boolean(activeMaskFrame?.maskIsInstance);
    for (let index = 0; index < frame.display.length; index += 1) {
      const gray = frame.display[index];
      const label = mask?.[index] ?? 0;
      const masked = label > 0;
      if (masked) {
        currentForegroundPixels += 1;
      }
      const outputIndex = index * 4;
      if (maskVisible && masked) {
        const color = maskIsInstance ? labelColor(label) : [255, 132, 32];
        output[outputIndex] = Math.round(gray * 0.35 + color[0] * 0.65);
        output[outputIndex + 1] = Math.round(gray * 0.35 + color[1] * 0.65);
        output[outputIndex + 2] = Math.round(gray * 0.35 + color[2] * 0.65);
      } else {
        output[outputIndex] = gray;
        output[outputIndex + 1] = gray;
        output[outputIndex + 2] = gray;
      }
      output[outputIndex + 3] = 255;
    }

    context.putImageData(imageData, 0, 0);
    setForegroundPixels(currentForegroundPixels);
  }, [activeMaskFrame, frame, maskVisible]);

  async function computeDataPreview() {
    if (!enabled || !inputPath || !source || !selectedTimepoint || !selectedTimepointMetadata || !processData) {
      return;
    }

    const selectedSource = source;
    const requestKey = requestedDataKey;
    setDataComputeLoading(true);
    setDataComputeError(null);

    try {
      const requestBody = {
        input_path: inputPath,
        gpu_index: gpuIndex,
        source_id: selectedSource.id,
        threshold_component_index: clampedThresholdComponent,
        data_operations: previewDataOperations,
        require_cached_data: false,
        timepoint: selectedTimepoint,
      };
      const resp = await fetch("/api/post-processing/general-segmentation/preview/data", {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });
      if (!resp.ok) {
        const detail = await safeJson(resp);
        const message =
          detail && typeof detail === "object" && "detail" in detail && typeof detail.detail === "string"
            ? detail.detail
            : `Preview data processing failed: ${resp.status} ${resp.statusText}`;
        throw new Error(message);
      }

      const json = (await resp.json()) as GeneralSegmentationPreviewDataResponse;
      setComputedDataKey(requestKey);
      setThresholdRange({
        key: requestKey,
        min: json.rangeMin,
        max: json.rangeMax,
        step: json.step,
      });
    } catch (error) {
      setDataComputeError(error instanceof Error ? error.message : "Unknown preview data processing error");
    } finally {
      setDataComputeLoading(false);
    }
  }

  async function computeProcessedMaskPreview() {
    if (!enabled || !inputPath || !source || !selectedTimepoint || !selectedTimepointMetadata || !processMask) {
      return;
    }

    const selectedSource = source;
    const requestKey = processedMaskKey;
    setMaskComputeLoading(true);
    setMaskComputeError(null);

    try {
      const requestBody = {
        input_path: inputPath,
        gpu_index: gpuIndex,
        source_id: selectedSource.id,
        threshold_component_index: clampedThresholdComponent,
        threshold: effectiveThresholdValue,
        invert_mask: invertMask,
        data_operations: activeDataOperations,
        require_cached_data: activeDataOperations.length > 0,
        mask_operations: previewMaskOperations,
        instance_method: "none",
        voronoi_spot_sigma: 0,
        voronoi_outline_sigma: 0,
        timepoint: selectedTimepoint,
        view: selectedView,
        z_index: currentZIndex,
      };
      const resp = await fetch("/api/post-processing/general-segmentation/preview/instance", {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });
      if (!resp.ok) {
        const detail = await safeJson(resp);
        const message =
          detail && typeof detail === "object" && "detail" in detail && typeof detail.detail === "string"
            ? detail.detail
            : `Preview mask processing failed: ${resp.status} ${resp.statusText}`;
        throw new Error(message);
      }

      const json = (await resp.json()) as GeneralSegmentationPreviewMaskResponse;
      const mask = decodeUint32Base64(json.mask.data);
      const pixelCount = json.width * json.height;
      if (mask.length !== pixelCount) {
        throw new Error("Preview processed mask payload size does not match its dimensions.");
      }
      setProcessedMaskFrame({
        key: requestKey,
        evaluationKey: activeEvaluationKey,
        timepoint: json.timepoint,
        sourceId: json.sourceId,
        view: json.view,
        zIndex: json.zIndex,
        width: json.width,
        height: json.height,
        shape: json.shape,
        mask,
        maskIsInstance: json.mask.instance,
      });
    } catch (error) {
      setMaskComputeError(error instanceof Error ? error.message : "Unknown preview mask processing error");
    } finally {
      setMaskComputeLoading(false);
    }
  }

  async function computeInstancePreview() {
    if (!enabled || !inputPath || !source || !selectedTimepoint || !selectedTimepointMetadata || instanceMethod === "none") {
      return;
    }

    const selectedSource = source;
    const requestKey = instanceMaskKey;
    setInstanceLoading(true);
    setInstanceError(null);

    try {
      const requestBody = {
        input_path: inputPath,
        gpu_index: gpuIndex,
        source_id: selectedSource.id,
        threshold_component_index: clampedThresholdComponent,
        threshold: effectiveThresholdValue,
        invert_mask: invertMask,
        data_operations: activeDataOperations,
        require_cached_data: activeDataOperations.length > 0,
        mask_operations: previewMaskOperations,
        instance_method: instanceMethod,
        voronoi_spot_sigma: previewVoronoiSpotSigma,
        voronoi_outline_sigma: previewVoronoiOutlineSigma,
        distance_transform_dynamic: previewDistanceDynamic,
        distance_transform_connectivity: distanceConnectivity,
        distance_transform_spacing_z: previewDistanceSpacingZ,
        distance_transform_spacing_y: previewDistanceSpacingY,
        distance_transform_spacing_x: previewDistanceSpacingX,
        intensity_prominence: previewIntensityProminence,
        intensity_smoothing_sigma: previewIntensitySmoothingSigma,
        intensity_low_percentile: previewIntensityLowPercentile,
        intensity_high_percentile: previewIntensityHighPercentile,
        intensity_connectivity: intensityConnectivity,
        timepoint: selectedTimepoint,
        view: selectedView,
        z_index: currentZIndex,
      };
      const resp = await fetch("/api/post-processing/general-segmentation/preview/instance", {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });
      if (!resp.ok) {
        const detail = await safeJson(resp);
        const message =
          detail && typeof detail === "object" && "detail" in detail && typeof detail.detail === "string"
            ? detail.detail
            : `Preview instance segmentation failed: ${resp.status} ${resp.statusText}`;
        throw new Error(message);
      }

      const json = (await resp.json()) as GeneralSegmentationPreviewMaskResponse;
      const mask = decodeUint32Base64(json.mask.data);
      const pixelCount = json.width * json.height;
      if (mask.length !== pixelCount) {
        throw new Error("Preview instance payload size does not match its dimensions.");
      }
      setInstanceMaskFrame({
        key: requestKey,
        evaluationKey: activeEvaluationKey,
        timepoint: json.timepoint,
        sourceId: json.sourceId,
        view: json.view,
        zIndex: json.zIndex,
        width: json.width,
        height: json.height,
        shape: json.shape,
        mask,
        maskIsInstance: json.mask.instance,
      });
    } catch (error) {
      setInstanceError(error instanceof Error ? error.message : "Unknown preview instance error");
    } finally {
      setInstanceLoading(false);
    }
  }

  if (!enabled) return null;

  const rangeLabel =
    thresholdRange?.key === activeDataKey
      ? `Range ${formatThresholdValue(thresholdMin, thresholdStep)}..${formatThresholdValue(thresholdMax, thresholdStep)}`
      : null;
  const updateDataOperation = (operationId: string, patch: Partial<GeneralSegmentationDataOperation>) => {
    onDataOperationsChange(
      dataOperations.map((operation) => (operation.id === operationId ? { ...operation, ...patch } : operation))
    );
  };
  const updateMaskOperation = (operationId: string, patch: Partial<GeneralSegmentationMaskOperation>) => {
    onMaskOperationsChange(
      maskOperations.map((operation) => (operation.id === operationId ? { ...operation, ...patch } : operation))
    );
  };
  const moveDataOperation = (operationId: string, direction: -1 | 1) => {
    const index = dataOperations.findIndex((operation) => operation.id === operationId);
    const nextIndex = index + direction;
    if (index < 0 || nextIndex < 0 || nextIndex >= dataOperations.length) return;
    const next = [...dataOperations];
    [next[index], next[nextIndex]] = [next[nextIndex], next[index]];
    onDataOperationsChange(next);
  };
  const moveMaskOperation = (operationId: string, direction: -1 | 1) => {
    const index = maskOperations.findIndex((operation) => operation.id === operationId);
    const nextIndex = index + direction;
    if (index < 0 || nextIndex < 0 || nextIndex >= maskOperations.length) return;
    const next = [...maskOperations];
    [next[index], next[nextIndex]] = [next[nextIndex], next[index]];
    onMaskOperationsChange(next);
  };
  const dataScalePatch = (
    key: "radius" | "sigma" | "sigma2",
    value: string,
    axis: "Z" | "Y" | "X" | null = null
  ): Partial<GeneralSegmentationDataOperation> => {
    if (key === "radius") {
      if (axis === "Z") return { radiusZ: value };
      if (axis === "Y") return { radiusY: value };
      if (axis === "X") return { radiusX: value };
      return { radius: value };
    }
    if (key === "sigma") {
      if (axis === "Z") return { sigmaZ: value };
      if (axis === "Y") return { sigmaY: value };
      if (axis === "X") return { sigmaX: value };
      return { sigma: value };
    }
    if (axis === "Z") return { sigma2Z: value };
    if (axis === "Y") return { sigma2Y: value };
    if (axis === "X") return { sigma2X: value };
    return { sigma2: value };
  };
  const renderDataAnisotropyToggle = (operation: GeneralSegmentationDataOperation) => (
    <label className="inferenceCheckboxLabel">
      <input
        type="checkbox"
        checked={operation.anisotropic}
        onChange={(event) => updateDataOperation(operation.id, { anisotropic: event.target.checked })}
      />
      <span>Anisotropic</span>
    </label>
  );
  const renderDataScaleInputs = (
    operation: GeneralSegmentationDataOperation,
    key: "radius" | "sigma" | "sigma2",
    label: string
  ) =>
    operation.anisotropic ? (
      <>
        <div className="inferenceInlineLabel isStrong">{`${label} Z`}</div>
        <PostProcessingNumberInput
          value={dataScaleAxisValue(operation, key, "Z")}
          onChange={(value) => updateDataOperation(operation.id, dataScalePatch(key, value, "Z"))}
          min={0}
          step={0.1}
        />
        <div className="inferenceInlineLabel isStrong">{`${label} Y`}</div>
        <PostProcessingNumberInput
          value={dataScaleAxisValue(operation, key, "Y")}
          onChange={(value) => updateDataOperation(operation.id, dataScalePatch(key, value, "Y"))}
          min={0}
          step={0.1}
        />
        <div className="inferenceInlineLabel isStrong">{`${label} X`}</div>
        <PostProcessingNumberInput
          value={dataScaleAxisValue(operation, key, "X")}
          onChange={(value) => updateDataOperation(operation.id, dataScalePatch(key, value, "X"))}
          min={0}
          step={0.1}
        />
      </>
    ) : (
      <>
        <div className="inferenceInlineLabel isStrong">{label}</div>
        <PostProcessingNumberInput
          value={dataScaleValue(operation, key)}
          onChange={(value) => updateDataOperation(operation.id, dataScalePatch(key, value))}
          min={0}
          step={0.1}
        />
      </>
    );
  const renderResponseSelect = (operation: GeneralSegmentationDataOperation) => (
    <>
      <div className="inferenceInlineLabel isStrong">Response</div>
      <select
        className="inferenceSelect inferenceCompactSelect"
        value={operation.response}
        onChange={(event) =>
          updateDataOperation(operation.id, {
            response: event.target.value as GeneralSegmentationLogResponse,
          })
        }
      >
        <option value="bright">Bright blobs</option>
        <option value="dark">Dark blobs</option>
      </select>
    </>
  );

  return (
    <div className="probabilityMapPreview">
      <div className="postProcessingStageTitle">General segmentation preview</div>

      {metadataLoading ? <ValidationMessage tone="neutral">Loading preview metadata...</ValidationMessage> : null}
      {metadataError ? <ValidationMessage tone="error">{metadataError}</ValidationMessage> : null}
      {!metadataLoading && !metadataError && metadata && compatibleTimepoints.length === 0 ? (
        <ValidationMessage tone="error">No previewable timepoints are available.</ValidationMessage>
      ) : null}

      {compatibleTimepoints.length > 0 ? (
        <>
          <div className="probabilityMapPreviewControls">
            <div className="inferenceFormRow">
              <div className="inferenceFieldLabel">Timepoint</div>
              <select
                className="inferenceSelect"
                value={selectedTimepoint}
                onChange={(event) => setSelectedTimepoint(event.target.value)}
              >
                {compatibleTimepoints.map((timepoint) => (
                  <option key={timepoint.name} value={timepoint.name}>
                    {timepoint.name}
                  </option>
                ))}
              </select>
              <div className="inferenceInlineLabel">
                {selectedTimepointMetadata
                  ? `${selectedTimepointMetadata.width} x ${selectedTimepointMetadata.height} x ${selectedTimepointMetadata.zCount}`
                  : null}
              </div>
            </div>

            <div className="inferenceFormRow">
              <div className="inferenceFieldLabel">
                <ParameterHelpLabel
                  label="Evaluation data"
                  description={POST_PROCESSING_PARAMETER_HELP.generalEvaluationData}
                />
              </div>
              <select
                className="inferenceSelect inferenceCompactSelect"
                value={source?.id ?? sourceId}
                onChange={(event) => onSourceIdChange(event.target.value)}
              >
                {sources.map((availableSource) => (
                  <option key={availableSource.id} value={availableSource.id}>
                    {availableSource.label}
                  </option>
                ))}
              </select>
              {hasSourceComponents ? (
                <select
                  className="inferenceSelect inferenceCompactSelect"
                  value={clampedThresholdComponent}
                  onChange={(event) =>
                    onThresholdComponentChange(clampInteger(Number.parseInt(event.target.value, 10), 0, maxComponent))
                  }
                >
                  {Array.from({ length: source.componentCount }, (_value, index) => (
                    <option key={index} value={index}>
                      {sourceComponentLabel(index)}
                    </option>
                  ))}
                </select>
              ) : null}
            </div>

            <div className="inferenceFormRow">
              <div className="inferenceFieldLabel">
                <ParameterHelpLabel label="Shown data" description={POST_PROCESSING_PARAMETER_HELP.generalDisplaySource} />
              </div>
              <select
                className="inferenceSelect inferenceCompactSelect"
                value={displaySource}
                onChange={(event) => onDisplaySourceChange(event.target.value as GeneralSegmentationDisplaySource)}
              >
                <option value="raw">Raw (unprocessed)</option>
                <option value="evaluation">Evaluation data</option>
              </select>
              <div className="inferenceInlineLabel isStrong">View</div>
              <select
                className="inferenceSelect inferenceCompactSelect"
                value={selectedView}
                onChange={(event) => setSelectedView(event.target.value as GeneralSegmentationPreviewView)}
              >
                <option value="slice">Z plane</option>
                <option value="max_projection">Max projection</option>
                <option value="min_projection">Min projection</option>
              </select>
              {selectedView === "slice" ? (
                <>
                  <input
                    type="range"
                    className="probabilityMapPreviewSlider"
                    min={0}
                    max={zMax}
                    step={1}
                    value={clampInteger(zIndex, 0, zMax)}
                    onChange={(event) => setZIndex(clampInteger(Number(event.target.value), 0, zMax))}
                    aria-label="Z plane"
                  />
                  <PostProcessingNumberInput
                    value={String(clampInteger(zIndex, 0, zMax))}
                    onChange={(value) => setZIndex(clampInteger(Number.parseInt(value, 10), 0, zMax))}
                    min={0}
                    max={zMax}
                    step={1}
                    ariaLabel="Z plane"
                  />
                </>
              ) : null}
              <div className="inferenceInlineLabel isStrong">Contrast</div>
              <select
                className="inferenceSelect inferenceCompactSelect"
                value={displayContrast}
                onChange={(event) => setDisplayContrast(event.target.value as GeneralSegmentationDisplayContrast)}
              >
                <option value="auto">Auto</option>
                <option value="full_range">Full range</option>
              </select>
            </div>

            <div className="inferenceFormRow">
              <div className="inferenceFieldLabel">
                <ParameterHelpLabel label="Process the data" description={POST_PROCESSING_PARAMETER_HELP.dataProcessing} />
              </div>
              <label className="inferenceCheckboxLabel">
                <input
                  type="checkbox"
                  checked={processData}
                  onChange={(event) => onProcessDataChange(event.target.checked)}
                />
                <span>Enabled</span>
              </label>
              {processData ? (
                <>
                  <button
                    type="button"
                    className="pickerSecondaryButton"
                    onClick={() => onDataOperationsChange([...dataOperations, createGeneralDataOperation()])}
                  >
                    Add operation
                  </button>
                  <button
                    type="button"
                    className="pickerSecondaryButton"
                    disabled={dataComputeLoading}
                    onClick={() => void computeDataPreview()}
                  >
                    {dataProcessingComputed ? "Recompute" : "Compute"}
                  </button>
                  {dataProcessState ? <div className="inferenceInlineLabel">{dataProcessState}</div> : null}
                </>
              ) : null}
            </div>

            {processData
              ? dataOperations.map((operation, index) => (
                  <div key={operation.id} className="inferenceFormRow">
                    <div className="inferenceFieldLabel">{`Data op ${index + 1}`}</div>
                    <select
                      className="inferenceSelect inferenceCompactSelect"
                      value={operation.type}
                      onChange={(event) =>
                        updateDataOperation(operation.id, {
                          type: event.target.value as GeneralSegmentationDataOperationType,
                        })
                      }
                    >
                      <option value="invert_lut">Invert LUT</option>
                      <option value="percentile_clipping">Percentile clip</option>
                      <option value="subtract_background">Subtract background</option>
                      <option value="gaussian_smoothing">Gaussian smoothing</option>
                      <option value="median_filter">Median filter</option>
                      <option value="laplacian_of_gaussian">Laplacian of Gaussian</option>
                      <option value="difference_of_gaussians">Difference of Gaussians</option>
                      <option value="top_hat">Top-hat</option>
                      <option value="black_hat">Black-hat</option>
                    </select>
                    {operation.type === "percentile_clipping" ? (
                      <>
                        <div className="inferenceInlineLabel isStrong">Low %</div>
                        <PostProcessingNumberInput
                          value={operation.lowPercentile}
                          onChange={(value) => updateDataOperation(operation.id, { lowPercentile: value })}
                          min={0}
                          max={100}
                          step={0.1}
                        />
                        <div className="inferenceInlineLabel isStrong">High %</div>
                        <PostProcessingNumberInput
                          value={operation.highPercentile}
                          onChange={(value) => updateDataOperation(operation.id, { highPercentile: value })}
                          min={0}
                          max={100}
                          step={0.1}
                        />
                        <label className="inferenceCheckboxLabel">
                          <input
                            type="checkbox"
                            checked={operation.rescale}
                            onChange={(event) => updateDataOperation(operation.id, { rescale: event.target.checked })}
                          />
                          <span>Rescale</span>
                        </label>
                        {operation.rescale ? (
                          <>
                            <div className="inferenceInlineLabel isStrong">Out min</div>
                            <PostProcessingNumberInput
                              value={operation.outputMin}
                              onChange={(value) => updateDataOperation(operation.id, { outputMin: value })}
                              step={0.1}
                            />
                            <div className="inferenceInlineLabel isStrong">Out max</div>
                            <PostProcessingNumberInput
                              value={operation.outputMax}
                              onChange={(value) => updateDataOperation(operation.id, { outputMax: value })}
                              step={0.1}
                            />
                          </>
                        ) : null}
                      </>
                    ) : null}
                    {operation.type === "subtract_background" ||
                    operation.type === "median_filter" ||
                    operation.type === "top_hat" ||
                    operation.type === "black_hat" ? (
                      <>
                        {renderDataAnisotropyToggle(operation)}
                        {renderDataScaleInputs(operation, "radius", "Radius")}
                      </>
                    ) : null}
                    {operation.type === "gaussian_smoothing" || operation.type === "laplacian_of_gaussian" ? (
                      <>
                        {renderDataAnisotropyToggle(operation)}
                        {renderDataScaleInputs(operation, "sigma", "Sigma")}
                      </>
                    ) : null}
                    {operation.type === "laplacian_of_gaussian" ? (
                      renderResponseSelect(operation)
                    ) : null}
                    {operation.type === "difference_of_gaussians" ? (
                      <>
                        {renderDataAnisotropyToggle(operation)}
                        {renderDataScaleInputs(operation, "sigma", "Small sigma")}
                        {renderDataScaleInputs(operation, "sigma2", "Large sigma")}
                        {renderResponseSelect(operation)}
                      </>
                    ) : null}
                    <button
                      type="button"
                      className="pickerSecondaryButton"
                      disabled={index === 0}
                      onClick={() => moveDataOperation(operation.id, -1)}
                    >
                      Up
                    </button>
                    <button
                      type="button"
                      className="pickerSecondaryButton"
                      disabled={index === dataOperations.length - 1}
                      onClick={() => moveDataOperation(operation.id, 1)}
                    >
                      Down
                    </button>
                    <button
                      type="button"
                      className="pickerSecondaryButton"
                      onClick={() => onDataOperationsChange(dataOperations.filter((item) => item.id !== operation.id))}
                    >
                      Remove
                    </button>
                  </div>
                ))
              : null}

            <div className="inferenceFormRow">
              <div className="inferenceFieldLabel">
                <ParameterHelpLabel label="Threshold" description={thresholdDescription} />
              </div>
              <input
                type="range"
                className="probabilityMapPreviewSlider"
                min={thresholdMin}
                max={thresholdMax}
                step={thresholdStep}
                value={sliderThresholdValue}
                onChange={(event) => onThresholdChange(event.target.value)}
                aria-label="Segmentation threshold"
              />
              <PostProcessingNumberInput
                value={threshold}
                onChange={onThresholdChange}
                min={thresholdMin}
                max={thresholdMax}
                step={thresholdStep}
                ariaLabel="Segmentation threshold"
              />
              <label className="inferenceCheckboxLabel">
                <input
                  type="checkbox"
                  checked={invertMask}
                  onChange={(event) => onInvertMaskChange(event.target.checked)}
                />
                <span>
                  <ParameterHelpLabel label="Invert mask" description={POST_PROCESSING_PARAMETER_HELP.invertMask} />
                </span>
              </label>
              <button
                type="button"
                className="pickerSecondaryButton"
                onClick={() => setMaskVisible((current) => !current)}
              >
                {maskVisible ? "Hide mask" : "Show mask"}
              </button>
              {frame ? (
                <div className="inferenceInlineLabel">{`Foreground ${foregroundPercent.toFixed(1)}%`}</div>
              ) : null}
              {rangeLoading ? <div className="inferenceInlineLabel">Updating range...</div> : null}
              {rangeLabel ? <div className="inferenceInlineLabel">{rangeLabel}</div> : null}
            </div>

            <div className="inferenceFormRow">
              <div className="inferenceFieldLabel">
                <ParameterHelpLabel label="Process the mask" description={POST_PROCESSING_PARAMETER_HELP.maskProcessing} />
              </div>
              <label className="inferenceCheckboxLabel">
                <input
                  type="checkbox"
                  checked={processMask}
                  onChange={(event) => onProcessMaskChange(event.target.checked)}
                />
                <span>Enabled</span>
              </label>
              {processMask ? (
                <>
                  <button
                    type="button"
                    className="pickerSecondaryButton"
                    onClick={() => onMaskOperationsChange([...maskOperations, createGeneralMaskOperation()])}
                  >
                    Add operation
                  </button>
                  <button
                    type="button"
                    className="pickerSecondaryButton"
                    disabled={maskComputeLoading || !frame}
                    onClick={() => void computeProcessedMaskPreview()}
                  >
                    {processedMaskFrame?.key === processedMaskKey ? "Recompute" : "Compute"}
                  </button>
                  {maskProcessState ? <div className="inferenceInlineLabel">{maskProcessState}</div> : null}
                </>
              ) : null}
            </div>

            {processMask
              ? maskOperations.map((operation, index) => (
                  <div key={operation.id} className="inferenceFormRow">
                    <div className="inferenceFieldLabel">{`Mask op ${index + 1}`}</div>
                    <select
                      className="inferenceSelect inferenceCompactSelect"
                      value={operation.type}
                      onChange={(event) =>
                        updateMaskOperation(operation.id, {
                          type: event.target.value as GeneralSegmentationMaskOperationType,
                        })
                      }
                    >
                      <option value="remove_small_objects">Remove small objects</option>
                      <option value="fill_small_holes">Fill small holes</option>
                      <option value="binary_closing">Binary closing</option>
                      <option value="binary_opening">Binary opening</option>
                      <option value="dilate">Dilate</option>
                      <option value="erode">Erode</option>
                      <option value="remove_border_objects">Remove border objects</option>
                      <option value="size_range">Size range filter</option>
                    </select>
                    {operation.type === "remove_small_objects" || operation.type === "fill_small_holes" ? (
                      <>
                        <div className="inferenceInlineLabel isStrong">Size</div>
                        <PostProcessingNumberInput
                          value={operation.size}
                          onChange={(value) => updateMaskOperation(operation.id, { size: value })}
                          min={0}
                          step={1}
                        />
                        <div className="inferenceInlineLabel">voxels</div>
                      </>
                    ) : null}
                    {operation.type === "binary_closing" ||
                    operation.type === "binary_opening" ||
                    operation.type === "dilate" ||
                    operation.type === "erode" ? (
                      <>
                        <div className="inferenceInlineLabel isStrong">Radius</div>
                        <PostProcessingNumberInput
                          value={operation.radius}
                          onChange={(value) => updateMaskOperation(operation.id, { radius: value })}
                          min={0}
                          step={0.1}
                        />
                        <div className="inferenceInlineLabel">voxels</div>
                      </>
                    ) : null}
                    {operation.type === "size_range" ? (
                      <>
                        <div className="inferenceInlineLabel isStrong">Min</div>
                        <PostProcessingNumberInput
                          value={operation.minSize}
                          onChange={(value) => updateMaskOperation(operation.id, { minSize: value })}
                          min={0}
                          step={1}
                        />
                        <div className="inferenceInlineLabel isStrong">Max</div>
                        <PostProcessingNumberInput
                          value={operation.maxSize}
                          onChange={(value) => updateMaskOperation(operation.id, { maxSize: value })}
                          min={0}
                          step={1}
                        />
                        <div className="inferenceInlineLabel">0 = no max</div>
                      </>
                    ) : null}
                    <button
                      type="button"
                      className="pickerSecondaryButton"
                      disabled={index === 0}
                      onClick={() => moveMaskOperation(operation.id, -1)}
                    >
                      Up
                    </button>
                    <button
                      type="button"
                      className="pickerSecondaryButton"
                      disabled={index === maskOperations.length - 1}
                      onClick={() => moveMaskOperation(operation.id, 1)}
                    >
                      Down
                    </button>
                    <button
                      type="button"
                      className="pickerSecondaryButton"
                      onClick={() => onMaskOperationsChange(maskOperations.filter((item) => item.id !== operation.id))}
                    >
                      Remove
                    </button>
                  </div>
                ))
              : null}

            <div className="inferenceFormRow">
              <div className="inferenceFieldLabel">
                <ParameterHelpLabel
                  label="Instance segmentation"
                  description={POST_PROCESSING_PARAMETER_HELP.instanceSegmentation}
                />
              </div>
              <select
                className="inferenceSelect inferenceCompactSelect"
                value={instanceMethod}
                onChange={(event) => onInstanceMethodChange(event.target.value as GeneralSegmentationInstanceMethod)}
              >
                <option value="none">None</option>
                <option value="connected_components">Connected-component labelling</option>
                <option value="voronoi_otsu">Voronoi-Otsu</option>
                <option value="distance_transform_watershed">Distance-transform watershed</option>
                <option value="intensity_prominence_watershed">Intensity-prominence watershed</option>
              </select>
              {instanceMethod === "voronoi_otsu" ? (
                <>
                  <div className="inferenceInlineLabel isStrong">Spot sigma</div>
                  <PostProcessingNumberInput
                    value={voronoiSpotSigma}
                    onChange={onVoronoiSpotSigmaChange}
                    min={0}
                    step={0.1}
                  />
                  <div className="inferenceInlineLabel isStrong">Outline sigma</div>
                  <PostProcessingNumberInput
                    value={voronoiOutlineSigma}
                    onChange={onVoronoiOutlineSigmaChange}
                    min={0}
                    step={0.1}
                  />
                </>
              ) : null}
              {instanceMethod === "distance_transform_watershed" ? (
                <>
                  <div className="inferenceInlineLabel isStrong">Dynamic</div>
                  <PostProcessingNumberInput
                    value={distanceDynamic}
                    onChange={onDistanceDynamicChange}
                    min={0}
                    step={0.1}
                  />
                  <div className="inferenceInlineLabel isStrong">Connectivity</div>
                  <select
                    className="inferenceSelect inferenceCompactSelect"
                    value={distanceConnectivity}
                    onChange={(event) =>
                      onDistanceConnectivityChange(Number.parseInt(event.target.value, 10) === 26 ? 26 : 6)
                    }
                  >
                    <option value={6}>6</option>
                    <option value={26}>26</option>
                  </select>
                  <div className="inferenceInlineLabel isStrong">Z spacing</div>
                  <PostProcessingNumberInput
                    value={distanceSpacingZ}
                    onChange={onDistanceSpacingZChange}
                    min={0.000001}
                    step={0.1}
                  />
                  <div className="inferenceInlineLabel isStrong">Y spacing</div>
                  <PostProcessingNumberInput
                    value={distanceSpacingY}
                    onChange={onDistanceSpacingYChange}
                    min={0.000001}
                    step={0.1}
                  />
                  <div className="inferenceInlineLabel isStrong">X spacing</div>
                  <PostProcessingNumberInput
                    value={distanceSpacingX}
                    onChange={onDistanceSpacingXChange}
                    min={0.000001}
                    step={0.1}
                  />
                </>
              ) : null}
              {instanceMethod === "intensity_prominence_watershed" ? (
                <>
                  <div className="inferenceInlineLabel isStrong">Prominence</div>
                  <PostProcessingNumberInput
                    value={intensityProminence}
                    onChange={onIntensityProminenceChange}
                    min={0}
                    max={1}
                    step={0.01}
                  />
                  <div className="inferenceInlineLabel isStrong">Smoothing sigma</div>
                  <PostProcessingNumberInput
                    value={intensitySmoothingSigma}
                    onChange={onIntensitySmoothingSigmaChange}
                    min={0}
                    step={0.1}
                  />
                  <div className="inferenceInlineLabel isStrong">Low %</div>
                  <PostProcessingNumberInput
                    value={intensityLowPercentile}
                    onChange={onIntensityLowPercentileChange}
                    min={0}
                    max={100}
                    step={0.1}
                  />
                  <div className="inferenceInlineLabel isStrong">High %</div>
                  <PostProcessingNumberInput
                    value={intensityHighPercentile}
                    onChange={onIntensityHighPercentileChange}
                    min={0}
                    max={100}
                    step={0.1}
                  />
                  <div className="inferenceInlineLabel isStrong">Connectivity</div>
                  <select
                    className="inferenceSelect inferenceCompactSelect"
                    value={intensityConnectivity}
                    onChange={(event) =>
                      onIntensityConnectivityChange(Number.parseInt(event.target.value, 10) === 26 ? 26 : 6)
                    }
                  >
                    <option value={6}>6</option>
                    <option value={26}>26</option>
                  </select>
                </>
              ) : null}
              {instanceMethod !== "none" ? (
                <>
                  <button
                    type="button"
                    className="pickerSecondaryButton"
                    disabled={instanceLoading || !frame}
                    onClick={() => void computeInstancePreview()}
                  >
                    {instanceMaskFrame?.key === instanceMaskKey ? "Recompute" : "Compute"}
                  </button>
                  {instancePreviewState ? <div className="inferenceInlineLabel">{instancePreviewState}</div> : null}
                </>
              ) : null}
            </div>
          </div>

          <div className="probabilityMapPreviewCanvasFrame">
            <canvas ref={canvasRef} className="probabilityMapPreviewCanvas" aria-label="General segmentation preview" />
            {frameLoading && !frame ? <div className="probabilityMapPreviewOverlay">Loading preview...</div> : null}
            {frameError ? <div className="probabilityMapPreviewOverlay isError">{frameError}</div> : null}
            {!frameError && rangeError ? <div className="probabilityMapPreviewOverlay isError">{rangeError}</div> : null}
            {!frameError && !rangeError && dataComputeError ? (
              <div className="probabilityMapPreviewOverlay isError">{dataComputeError}</div>
            ) : null}
            {!frameError && maskError ? <div className="probabilityMapPreviewOverlay isError">{maskError}</div> : null}
            {!frameError && !maskError && maskComputeError ? (
              <div className="probabilityMapPreviewOverlay isError">{maskComputeError}</div>
            ) : null}
            {!frameError && !maskError && !maskComputeError && instanceError ? (
              <div className="probabilityMapPreviewOverlay isError">{instanceError}</div>
            ) : null}
            {!frameLoading && !frameError && !frame ? (
              <div className="probabilityMapPreviewOverlay">No preview loaded.</div>
            ) : null}
          </div>
        </>
      ) : null}
    </div>
  );
}

function DirectoryFieldRow({
  label,
  path,
  buttonLabel,
  emptyLabel,
  onChoose,
  action,
}: {
  label: ReactNode;
  path: string | null;
  buttonLabel: string;
  emptyLabel: string;
  onChoose: () => void;
  action?: ReactNode;
}) {
  return (
    <div className="inferencePathRow">
      <div className="inferencePathLabel">{label}</div>
      <button type="button" className="pickerPrimaryButton" onClick={onChoose}>
        {buttonLabel}
      </button>
      <div className={path ? "datasetPath" : "datasetPath isEmpty"}>
        <div className="datasetPathValue">{path ?? emptyLabel}</div>
      </div>
      {action ? <div className="inferencePathAction">{action}</div> : null}
    </div>
  );
}

function GpuSelectionRow({
  optionsLoading,
  availableGpus,
  gpuError,
  selectedGpuIndex,
  onSelectGpu,
  helpDescription,
}: {
  optionsLoading: boolean;
  availableGpus: GpuOption[];
  gpuError: string | null;
  selectedGpuIndex: number | null;
  onSelectGpu: (gpuIndex: number | null) => void;
  helpDescription: string;
}) {
  return (
    <div className="inferenceFormRow">
      <div className="inferenceFieldLabel">
        <ParameterHelpLabel label="Select GPU" description={helpDescription} />
      </div>
      <div className="inferenceCheckboxGroup">
        {optionsLoading ? <div className="sidebarHint">Loading GPUs...</div> : null}
        {!optionsLoading && availableGpus.length === 0 && !gpuError ? (
          <div className="sidebarHint">No NVIDIA GPUs detected.</div>
        ) : null}
        {availableGpus.map((gpu) => (
          <label key={gpu.index} className="inferenceCheckboxLabel">
            <input
              type="checkbox"
              checked={selectedGpuIndex === gpu.index}
              onChange={() => onSelectGpu(selectedGpuIndex === gpu.index ? null : gpu.index)}
            />
            <span>{`GPU ${gpu.index}`}</span>
          </label>
        ))}
        {gpuError ? <div className="sidebarError">{gpuError}</div> : null}
      </div>
    </div>
  );
}

function PostProcessingNumberInput({
  value,
  onChange,
  min,
  step,
  max,
  ariaLabel,
}: {
  value: string;
  onChange: (value: string) => void;
  min?: number;
  step: number;
  max?: number;
  ariaLabel?: string;
}) {
  return (
    <input
      type="number"
      className="inferenceNumberInput"
      value={value}
      onChange={(event) => onChange(event.target.value)}
      min={min}
      step={step}
      max={max}
      aria-label={ariaLabel}
    />
  );
}

function PostProcessingTextInput({
  value,
  onChange,
  placeholder,
  className,
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  className?: string;
}) {
  return (
    <input
      type="text"
      className={["inferenceNumberInput", className].filter(Boolean).join(" ")}
      value={value}
      onChange={(event) => onChange(event.target.value)}
      placeholder={placeholder}
    />
  );
}

function ValidationMessage({
  tone,
  children,
}: {
  tone: "neutral" | "success" | "error";
  children: string;
}) {
  return <div className={`inferenceValidationState is-${tone}`}>{children}</div>;
}

async function safeJson(resp: Response): Promise<unknown> {
  try {
    return await resp.json();
  } catch {
    return null;
  }
}

function parseNullableInteger(value: string): number | null {
  const trimmed = value.trim();
  if (!trimmed) return null;
  const parsed = Number.parseInt(trimmed, 10);
  return Number.isFinite(parsed) ? parsed : null;
}

function decodeBase64Bytes(value: string): Uint8Array {
  const binary = window.atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

function decodeFloat32Base64(value: string): Float32Array {
  const bytes = decodeBase64Bytes(value);
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  return new Float32Array(buffer);
}

function decodeUint32Base64(value: string): Uint32Array {
  const bytes = decodeBase64Bytes(value);
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  return new Uint32Array(buffer);
}

function labelColor(label: number): [number, number, number] {
  const hue = ((label * 137.508) % 360) / 360;
  return hsvToRgb(hue, 0.78, 1);
}

function hsvToRgb(hue: number, saturation: number, value: number): [number, number, number] {
  const sector = Math.floor(hue * 6);
  const fraction = hue * 6 - sector;
  const p = value * (1 - saturation);
  const q = value * (1 - fraction * saturation);
  const t = value * (1 - (1 - fraction) * saturation);
  const channelSets: Array<[number, number, number]> = [
    [value, t, p],
    [q, value, p],
    [p, value, t],
    [p, q, value],
    [t, p, value],
    [value, p, q],
  ];
  const [red, green, blue] = channelSets[sector % 6];
  return [Math.round(red * 255), Math.round(green * 255), Math.round(blue * 255)];
}

function formatThresholdValue(value: number, step: number): string {
  if (!Number.isFinite(value)) return "0";
  if (!Number.isFinite(step) || step >= 1) return String(Math.round(value));
  const decimals = Math.min(15, Math.max(0, Math.ceil(-Math.log10(step)) + 1));
  return value.toFixed(decimals).replace(/\.?0+$/, "");
}

function clampNumber(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return Number.isFinite(min) && Number.isFinite(max) ? (min + max) / 2 : 0;
  if (!Number.isFinite(min) || !Number.isFinite(max)) return value;
  const low = Math.min(min, max);
  const high = Math.max(min, max);
  return Math.min(high, Math.max(low, value));
}

function clampInteger(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, Math.round(value)));
}
