import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";

import ParameterHelpLabel from "../components/ParameterHelpLabel";
import { useJobs } from "../components/JobsProvider";
import ServerDirectoryPicker from "../components/ServerDirectoryPicker";
import { getClientId } from "../lib/clientId";

export type PostProcessingPageKind = "post_processing" | "segmentation" | "tracking";
type WorkflowOption = "process_features" | "foreground_probability_map" | "segmentation" | "tracking";
type PostProcessingMode = "pca" | "high_resolution_features" | "foreground_probability_map";
type SegmentationMode = "voronoi_otsu" | "probability_map" | "legacy_probability_map";
type SaveFormat = ".npy" | ".tif";

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
};

type SegmentationRunRequest = {
  input_path: string;
  output_path: string | null;
  densities_path: string | null;
  gpu_index: number | null;
  mode: SegmentationMode;
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

type ProbabilityMapPreviewView = "slice" | "max_projection";

type ProbabilityMapPreviewShape = {
  z: number;
  y: number;
  x: number;
};

type ProbabilityMapPreviewTimepoint = {
  name: string;
  compatible: boolean;
  message?: string;
  rawShape?: ProbabilityMapPreviewShape;
  probmapShape?: ProbabilityMapPreviewShape;
  shape?: ProbabilityMapPreviewShape;
  zCount?: number;
  width?: number;
  height?: number;
};

type ProbabilityMapPreviewMetadataResponse = {
  valid: boolean;
  message: string;
  reasonCode?: string;
  subfolderCount?: number;
  subfolderNames?: string[];
  timepoints?: ProbabilityMapPreviewTimepoint[];
  defaultTimepoint?: string | null;
};

type ProbabilityMapPreviewImageResponse = {
  timepoint: string;
  view: ProbabilityMapPreviewView;
  zIndex: number | null;
  width: number;
  height: number;
  shape: ProbabilityMapPreviewShape;
  raw: {
    dtype: "uint8";
    data: string;
    displayLow: number;
    displayHigh: number;
  };
  probability: {
    dtype: "float32";
    data: string;
  };
};

type ProbabilityMapPreviewFrame = {
  timepoint: string;
  view: ProbabilityMapPreviewView;
  zIndex: number | null;
  width: number;
  height: number;
  shape: ProbabilityMapPreviewShape;
  raw: Uint8Array;
  probability: Float32Array;
  rawDisplayLow: number;
  rawDisplayHigh: number;
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
    label: "Voronoi-Otsu",
  },
  {
    value: "probability_map",
    label: "Probability map",
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
    "Root folder where Process features saves pca_<n>/ and hr_feats/ outputs.",
  chosenFiles: "Limit processing to a contiguous range of validated timepoints. End file is inclusive.",
  segmentationOutputFolder:
    "Root folder where segmentation saves seg_voronoi/, seg_probmap/, seg_probmap_legacy/, and probmap_densities.npz.",
  foregroundProbabilityMapOutputFolder:
    "Root folder where foreground probability-map generation saves probmap/ and probmap_densities.npz.",
  trackingOutputFolder: "Folder where tracking saves the output CSV.",
  trackingOutputFilename: "CSV file name to write inside the selected output folder.",
  segmentationFolder: "Folder containing one segmentation mask per timepoint, named <timepoint>.tif.",
  selectGpu: "Choose the GPU that will run this post-processing step.",
  saveHighResolutionFeatures: "Write one full-resolution feature volume per channel under hr_feats/<timepoint>/.",
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
  simpleProbabilityThreshold: "Minimum normalized foreground probability used to mark voxels as foreground.",
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
  const [simpleProbabilityMapThreshold, setSimpleProbabilityMapThreshold] = useState("0.5");
  const [runProbabilityMapCcl, setRunProbabilityMapCcl] = useState(true);
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
  const voronoiOtsuSelected = segmentationSelected && selectedSegmentationMode === "voronoi_otsu";
  const probabilityMapSelected = segmentationSelected && selectedSegmentationMode === "probability_map";
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
    simpleProbabilityMapThreshold,
    runProbabilityMapCcl,
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
          : probabilityMapSelected
            ? "/api/post-processing/probability-map/validate-input"
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
      setRunFeedback({ tone: "error", message: "Enter a valid positive integer for the number of PCA components." });
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

    if (selectedSegmentationMode === "probability_map") {
      const parsedThreshold = Number.parseFloat(simpleProbabilityMapThreshold.trim());
      if (!Number.isFinite(parsedThreshold) || parsedThreshold < 0 || parsedThreshold > 1) {
        setRunFeedback({ tone: "error", message: "Enter a valid foreground probability threshold between 0 and 1." });
        return;
      }

      await submitRun("/api/post-processing/segmentation/run", {
        input_path: inputPath,
        output_path: effectiveSegmentationOutputPath,
        densities_path: null,
        gpu_index: null,
        mode: "probability_map",
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
        probmap_threshold: parsedThreshold,
        run_connected_components: runProbabilityMapCcl,
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

          {probabilityMapSelected ? (
            <div className="inferenceFormRows">
              <ProbabilityMapPreview
                inputPath={inputPath}
                enabled={probabilityMapSelected && parametersVisible}
                threshold={simpleProbabilityMapThreshold}
                onThresholdChange={setSimpleProbabilityMapThreshold}
                thresholdDescription={POST_PROCESSING_PARAMETER_HELP.simpleProbabilityThreshold}
              />

              <div className="inferenceFormRow">
                <div className="inferenceFieldLabel">
                  <ParameterHelpLabel
                    label="Run connected-component labeling"
                    description={POST_PROCESSING_PARAMETER_HELP.runConnectedComponents}
                  />
                </div>
                <label className="inferenceCheckboxLabel">
                  <input
                    type="checkbox"
                    checked={runProbabilityMapCcl}
                    onChange={(event) => setRunProbabilityMapCcl(event.target.checked)}
                  />
                  <span>Enabled</span>
                </label>
              </div>
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

function ProbabilityMapPreview({
  inputPath,
  enabled,
  threshold,
  onThresholdChange,
  thresholdDescription,
}: {
  inputPath: string | null;
  enabled: boolean;
  threshold: string;
  onThresholdChange: (value: string) => void;
  thresholdDescription: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [metadata, setMetadata] = useState<ProbabilityMapPreviewMetadataResponse | null>(null);
  const [metadataLoading, setMetadataLoading] = useState(false);
  const [metadataError, setMetadataError] = useState<string | null>(null);
  const [selectedTimepoint, setSelectedTimepoint] = useState("");
  const [selectedView, setSelectedView] = useState<ProbabilityMapPreviewView>("slice");
  const [zIndex, setZIndex] = useState(0);
  const [maskVisible, setMaskVisible] = useState(true);
  const [frame, setFrame] = useState<ProbabilityMapPreviewFrame | null>(null);
  const [frameLoading, setFrameLoading] = useState(false);
  const [frameError, setFrameError] = useState<string | null>(null);
  const [foregroundPixels, setForegroundPixels] = useState(0);

  const timepoints = metadata?.timepoints ?? [];
  const compatibleTimepoints = timepoints.filter((timepoint) => timepoint.compatible);
  const selectedTimepointMetadata =
    compatibleTimepoints.find((timepoint) => timepoint.name === selectedTimepoint) ?? null;
  const zMax = Math.max(0, (selectedTimepointMetadata?.zCount ?? 1) - 1);
  const thresholdValue = normalizeProbabilityThreshold(threshold);
  const totalPixels = frame ? frame.width * frame.height : 0;
  const foregroundPercent = totalPixels > 0 ? (foregroundPixels / totalPixels) * 100 : 0;

  useEffect(() => {
    if (!enabled || !inputPath) {
      setMetadata(null);
      setMetadataLoading(false);
      setMetadataError(null);
      setSelectedTimepoint("");
      setFrame(null);
      return;
    }

    const controller = new AbortController();
    setMetadataLoading(true);
    setMetadataError(null);
    setMetadata(null);
    setFrame(null);

    async function loadMetadata() {
      try {
        const resp = await fetch("/api/post-processing/probability-map/preview/metadata", {
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

        const json = (await resp.json()) as ProbabilityMapPreviewMetadataResponse;
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
    if (!enabled || !inputPath || !selectedTimepoint || !selectedTimepointMetadata) {
      setFrame(null);
      setFrameLoading(false);
      setFrameError(null);
      return;
    }

    const controller = new AbortController();
    setFrameLoading(true);
    setFrameError(null);

    async function loadFrame() {
      try {
        const requestBody = {
          input_path: inputPath,
          timepoint: selectedTimepoint,
          view: selectedView,
          z_index: selectedView === "slice" ? clampInteger(zIndex, 0, zMax) : null,
        };
        const resp = await fetch("/api/post-processing/probability-map/preview/image", {
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
              : `Preview image failed: ${resp.status} ${resp.statusText}`;
          throw new Error(message);
        }

        const json = (await resp.json()) as ProbabilityMapPreviewImageResponse;
        const raw = decodeBase64Bytes(json.raw.data);
        const probability = decodeFloat32Base64(json.probability.data);
        const pixelCount = json.width * json.height;
        if (raw.length !== pixelCount || probability.length !== pixelCount) {
          throw new Error("Preview payload size does not match its dimensions.");
        }
        setFrame({
          timepoint: json.timepoint,
          view: json.view,
          zIndex: json.zIndex,
          width: json.width,
          height: json.height,
          shape: json.shape,
          raw,
          probability,
          rawDisplayLow: json.raw.displayLow,
          rawDisplayHigh: json.raw.displayHigh,
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
  }, [enabled, inputPath, selectedTimepoint, selectedTimepointMetadata, selectedView, zIndex, zMax]);

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
    for (let index = 0; index < frame.raw.length; index += 1) {
      const gray = frame.raw[index];
      const masked = frame.probability[index] >= thresholdValue;
      if (masked) {
        currentForegroundPixels += 1;
      }
      const outputIndex = index * 4;
      if (maskVisible && masked) {
        output[outputIndex] = Math.round(gray * 0.35 + 255 * 0.65);
        output[outputIndex + 1] = Math.round(gray * 0.35 + 132 * 0.65);
        output[outputIndex + 2] = Math.round(gray * 0.35 + 32 * 0.65);
      } else {
        output[outputIndex] = gray;
        output[outputIndex + 1] = gray;
        output[outputIndex + 2] = gray;
      }
      output[outputIndex + 3] = 255;
    }

    context.putImageData(imageData, 0, 0);
    setForegroundPixels(currentForegroundPixels);
  }, [frame, maskVisible, thresholdValue]);

  if (!enabled) return null;

  return (
    <div className="probabilityMapPreview">
      <div className="postProcessingStageTitle">Probability map preview</div>

      {metadataLoading ? <ValidationMessage tone="neutral">Loading preview metadata...</ValidationMessage> : null}
      {metadataError ? <ValidationMessage tone="error">{metadataError}</ValidationMessage> : null}
      {!metadataLoading && !metadataError && metadata && compatibleTimepoints.length === 0 ? (
        <ValidationMessage tone="error">No previewable timepoints have matching raw and probability-map shapes.</ValidationMessage>
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
              <div className="inferenceFieldLabel">View</div>
              <select
                className="inferenceSelect inferenceCompactSelect"
                value={selectedView}
                onChange={(event) => setSelectedView(event.target.value as ProbabilityMapPreviewView)}
              >
                <option value="slice">Z plane</option>
                <option value="max_projection">Max projection</option>
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
            </div>

            <div className="inferenceFormRow">
              <div className="inferenceFieldLabel">
                <ParameterHelpLabel label="Threshold" description={thresholdDescription} />
              </div>
              <input
                type="range"
                className="probabilityMapPreviewSlider"
                min={0}
                max={1}
                step={0.01}
                value={thresholdValue}
                onChange={(event) => onThresholdChange(event.target.value)}
                aria-label="Foreground probability threshold"
              />
              <PostProcessingNumberInput
                value={threshold}
                onChange={onThresholdChange}
                min={0}
                max={1}
                step={0.01}
                ariaLabel="Foreground probability threshold"
              />
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
            </div>
          </div>

          <div className="probabilityMapPreviewCanvasFrame">
            <canvas ref={canvasRef} className="probabilityMapPreviewCanvas" aria-label="Probability map preview" />
            {frameLoading ? <div className="probabilityMapPreviewOverlay">Loading preview...</div> : null}
            {frameError ? <div className="probabilityMapPreviewOverlay isError">{frameError}</div> : null}
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
  min: number;
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

function normalizeProbabilityThreshold(value: string): number {
  const parsed = Number.parseFloat(value);
  if (!Number.isFinite(parsed)) return 0.5;
  return Math.min(1, Math.max(0, parsed));
}

function clampInteger(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, Math.round(value)));
}
