import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";

import ParameterHelpLabel from "../components/ParameterHelpLabel";
import { useJobs } from "../components/JobsProvider";
import ServerDirectoryPicker from "../components/ServerDirectoryPicker";
import { getClientId } from "../lib/clientId";

type WorkflowOption = "process_features" | "segmentation" | "tracking";
type SegmentationMode = "voronoi_otsu" | "probability_map";
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
  save_high_resolution_features: boolean;
  high_resolution_save_format: SaveFormat;
  save_pca: boolean;
  pca_components: number;
  pca_save_format: SaveFormat;
};

type SegmentationRunRequest = {
  input_path: string;
  output_path: string | null;
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
  seed: number;
};

type TrackingRunRequest = {
  input_path: string;
  segmentation_path: string;
  output_path: string;
  max_distance_xy: number;
  max_distance_z: number;
  z_distance_weight: number;
  min_distance_to_remove_cand: number;
  vote_thresholds: string;
  dice_threshold: number;
  corr_threshold: number;
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

const WORKFLOW_OPTIONS: Array<{
  value: WorkflowOption;
  label: string;
}> = [
  {
    value: "process_features",
    label: "Process features",
  },
  {
    value: "segmentation",
    label: "Segmentation",
  },
  {
    value: "tracking",
    label: "Tracking",
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
];

const INFERENCE_OUTPUT_FOLDER_DESCRIPTION = "folder containing the outputs of a SpatialDINO run";

const POST_PROCESSING_PARAMETER_HELP = {
  inferenceOutputFolder: INFERENCE_OUTPUT_FOLDER_DESCRIPTION,
  processFeaturesOutputFolder:
    "Folder where Process features saves PCA volumes and/or high-resolution features in one subfolder per timepoint.",
  segmentationOutputFolder:
    "Folder where segmentation masks are saved as one <timepoint>.tif file per validated subfolder.",
  trackingOutputFolder: "Folder where tracking saves tracks.csv.",
  segmentationFolder: "Folder containing one segmentation mask per timepoint, named <timepoint>.tif.",
  selectGpu: "Choose the GPU that will run this post-processing step.",
  saveHighResolutionFeatures: "Write one full-resolution feature volume per channel for each subfolder.",
  saveFormat: "Choose whether the generated outputs are saved as NumPy arrays or TIFF volumes.",
  savePca: "Export PCA-compressed feature volumes alongside the main outputs.",
  components: "Set how many principal components to keep in the PCA export.",
  voronoiOtsu: "Tune the classical Voronoi-Otsu segmentation pipeline.",
  gaussianBlurSigma: "Smooth the image before seed detection.",
  rollingBallRadius: "Set the background-removal scale used before thresholding.",
  densityEstimationToggle: "Run density estimation now instead of reusing an existing density file.",
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
  matchWindow: "Set the maximum per-frame search distance for track matching.",
  maxXyDistance: "Largest allowed XY displacement between linked detections.",
  maxZDistance: "Largest allowed Z displacement between linked detections.",
  distanceLogic: "Weight and shortcut rules used when resolving candidate matches.",
  zDistanceWeight: "Multiplier applied to Z distance when scoring matches.",
  immediateAssignmentDistance: "Distance below which a candidate is assigned immediately.",
  votingThresholds: "Tune the overlap-and-voting acceptance rules for track linking.",
  voteThresholds: "Comma-separated vote cutoffs checked in order.",
  minDice: "Minimum Dice overlap required for a match.",
  minCorr: "Minimum intensity correlation required for a match.",
} as const;

function getWorkflowLabel(workflow: WorkflowOption | null): string {
  if (!workflow) return "Post-processing";
  return WORKFLOW_OPTIONS.find((option) => option.value === workflow)?.label ?? "Post-processing";
}

function parentPathForPicker(path: string | null): string | null {
  if (!path) return null;
  const normalized = path.trim();
  if (!normalized.startsWith("/")) return null;
  const slashIndex = normalized.lastIndexOf("/");
  if (slashIndex <= 0) return "/";
  return normalized.slice(0, slashIndex);
}

export default function PostProcessingPage() {
  const jobs = useJobs();
  const validationRequestIdRef = useRef(0);
  const trackingSegmentationValidationRequestIdRef = useRef(0);

  const [selectedWorkflow, setSelectedWorkflow] = useState<WorkflowOption | null>(null);
  const [selectedSegmentationMode, setSelectedSegmentationMode] = useState<SegmentationMode>("voronoi_otsu");
  const [pickerOpen, setPickerOpen] = useState(false);
  const [pickerTarget, setPickerTarget] = useState<
    "input" | "process_output" | "segmentation_output" | "tracking_segmentation" | "tracking_output" | "seg_tif" | "valid_mask"
  >("input");
  const [inputPath, setInputPath] = useState<string | null>(null);
  const [processFeaturesOutputPath, setProcessFeaturesOutputPath] = useState<string | null>(null);
  const [segmentationOutputPath, setSegmentationOutputPath] = useState<string | null>(null);
  const [trackingSegmentationPath, setTrackingSegmentationPath] = useState<string | null>(null);
  const [trackingOutputPath, setTrackingOutputPath] = useState<string | null>(null);
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
  const [saveHighResolutionFeatures, setSaveHighResolutionFeatures] = useState(false);
  const [highResolutionSaveFormat, setHighResolutionSaveFormat] = useState<SaveFormat>(".tif");
  const [savePca, setSavePca] = useState(false);
  const [pcaComponents, setPcaComponents] = useState("3");
  const [pcaSaveFormat, setPcaSaveFormat] = useState<SaveFormat>(".tif");
  const [gaussianBlurSigma, setGaussianBlurSigma] = useState("3");
  const [rollingBallRadius, setRollingBallRadius] = useState("10");
  const [runDensityEstimation, setRunDensityEstimation] = useState(true);
  const [runProbabilityMapStage2, setRunProbabilityMapStage2] = useState(true);
  const [densityEstimationTimepoint, setDensityEstimationTimepoint] = useState<string>("");
  const [segTifPath, setSegTifPath] = useState<string | null>(null);
  const [validMaskTifPath, setValidMaskTifPath] = useState<string | null>(null);
  const [probabilityMapDensityMethod, setProbabilityMapDensityMethod] = useState<"gpu-hist" | "kde">("gpu-hist");
  const [probabilityMapFeatureBatch, setProbabilityMapFeatureBatch] = useState("32");
  const [probabilityMapKdePoints, setProbabilityMapKdePoints] = useState("512");
  const [probabilityMapKdeMaxSamples, setProbabilityMapKdeMaxSamples] = useState("200000");
  const [probabilityMapKdeBandwidth, setProbabilityMapKdeBandwidth] = useState("");
  const [probabilityMapHistSigmaBins, setProbabilityMapHistSigmaBins] = useState("1.5");
  const [probabilityMapBgThreshold, setProbabilityMapBgThreshold] = useState("0.4");
  const [probabilityMapFgThreshold, setProbabilityMapFgThreshold] = useState("0.95");
  const [probabilityMapSeed, setProbabilityMapSeed] = useState("1337");
  const [trackingMaxDistanceXy, setTrackingMaxDistanceXy] = useState("20");
  const [trackingMaxDistanceZ, setTrackingMaxDistanceZ] = useState("10");
  const [trackingZDistanceWeight, setTrackingZDistanceWeight] = useState("2.5");
  const [trackingMinDistanceToRemoveCand, setTrackingMinDistanceToRemoveCand] = useState("3");
  const [trackingVoteThresholds, setTrackingVoteThresholds] = useState("320,300,280,260");
  const [trackingDiceThreshold, setTrackingDiceThreshold] = useState("0.5");
  const [trackingCorrThreshold, setTrackingCorrThreshold] = useState("0.5");
  const [voronoiOptionalOpen, setVoronoiOptionalOpen] = useState(false);
  const [probabilityMapOptionalOpen, setProbabilityMapOptionalOpen] = useState(false);
  const [trackingOptionalOpen, setTrackingOptionalOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [runFeedback, setRunFeedback] = useState<RunFeedback | null>(null);

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
    trackingSegmentationValidationRequestIdRef.current += 1;
    setTrackingSegmentationValidationLoading(false);
    setTrackingSegmentationValidationResult(null);
  }, [inputPath, trackingSegmentationPath, selectedWorkflow]);

  useEffect(() => {
    setRunFeedback(null);
  }, [
    selectedWorkflow,
    selectedSegmentationMode,
    inputPath,
    processFeaturesOutputPath,
    segmentationOutputPath,
    trackingSegmentationPath,
    trackingOutputPath,
    selectedGpuIndex,
    saveHighResolutionFeatures,
    highResolutionSaveFormat,
    savePca,
    pcaComponents,
    pcaSaveFormat,
    gaussianBlurSigma,
    rollingBallRadius,
    runDensityEstimation,
    densityEstimationTimepoint,
    segTifPath,
    validMaskTifPath,
    probabilityMapDensityMethod,
    probabilityMapFeatureBatch,
    probabilityMapKdePoints,
    probabilityMapKdeMaxSamples,
    probabilityMapKdeBandwidth,
    probabilityMapHistSigmaBins,
    probabilityMapBgThreshold,
    probabilityMapFgThreshold,
    probabilityMapSeed,
    trackingMaxDistanceXy,
    trackingMaxDistanceZ,
    trackingZDistanceWeight,
    trackingMinDistanceToRemoveCand,
    trackingVoteThresholds,
    trackingDiceThreshold,
    trackingCorrThreshold,
  ]);

  const workflowLabel = getWorkflowLabel(selectedWorkflow);
  const inputStepVisible = selectedWorkflow !== null;
  const processFeaturesSelected = selectedWorkflow === "process_features";
  const segmentationSelected = selectedWorkflow === "segmentation";
  const trackingSelected = selectedWorkflow === "tracking";
  const voronoiOtsuSelected = segmentationSelected && selectedSegmentationMode === "voronoi_otsu";
  const probabilityMapSelected = segmentationSelected && selectedSegmentationMode === "probability_map";
  const inputValidated = validationResult?.valid === true;
  const trackingSegmentationValidated = trackingSegmentationValidationResult?.valid === true;
  const parametersVisible = trackingSelected ? inputValidated && trackingSegmentationValidated : inputValidated;
  const validatedSubfolderNames = inputValidated ? validationResult.subfolderNames ?? [] : [];
  const showDensityEstimationTimepoint = validatedSubfolderNames.length > 1;
  const probmapDensitiesPath = inputValidated ? validationResult.probmapDensitiesPath ?? null : null;
  const probmapDensitiesExists = inputValidated ? Boolean(validationResult.probmapDensitiesExists) : false;
  const inputValidationSuccessMessage = validationResult?.valid
    ? `Validated inference output folder. Found ${validationResult.subfolderCount} subfolder${
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
        : pickerTarget === "segmentation_output" || pickerTarget === "tracking_segmentation"
          ? "Choose the segmentation output folder"
          : pickerTarget === "seg_tif"
            ? "Choose the annotated FG/BG mask"
            : "Choose the valid voxels mask";
  const pickerInitialPath =
    pickerTarget === "input"
      ? inputPath
      : pickerTarget === "process_output"
        ? processFeaturesOutputPath ?? inputPath
        : pickerTarget === "segmentation_output"
          ? segmentationOutputPath
          : pickerTarget === "tracking_segmentation"
            ? trackingSegmentationPath ?? inputPath
            : pickerTarget === "tracking_output"
              ? trackingOutputPath ?? trackingSegmentationPath ?? inputPath
              : pickerTarget === "seg_tif"
                ? parentPathForPicker(segTifPath) ?? inputPath
                : parentPathForPicker(validMaskTifPath) ?? inputPath;

  useEffect(() => {
    if (!probabilityMapSelected) return;
    if (!validatedSubfolderNames.length) return;
    if (densityEstimationTimepoint && validatedSubfolderNames.includes(densityEstimationTimepoint)) return;
    setDensityEstimationTimepoint(validatedSubfolderNames[0]);
  }, [densityEstimationTimepoint, probabilityMapSelected, validatedSubfolderNames]);

  async function validateInputFolder() {
    if (!inputPath) return;
    if (!selectedWorkflow) return;

    const requestId = validationRequestIdRef.current + 1;
    validationRequestIdRef.current = requestId;
    setValidationLoading(true);
    setValidationResult(null);

    try {
      const validationUrl =
        selectedWorkflow === "tracking"
          ? "/api/post-processing/tracking/validate-input"
          : selectedWorkflow === "segmentation"
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
    if (!inputPath || !processFeaturesOutputPath) return null;

    const parsedPcaComponents = Number.parseInt(pcaComponents.trim(), 10);
    if (savePca && (!Number.isFinite(parsedPcaComponents) || parsedPcaComponents < 1)) {
      return null;
    }

    return {
      input_path: inputPath,
      output_path: processFeaturesOutputPath,
      gpu_index: selectedGpuIndex,
      save_high_resolution_features: saveHighResolutionFeatures,
      high_resolution_save_format: highResolutionSaveFormat,
      save_pca: savePca,
      pca_components: Number.isFinite(parsedPcaComponents) && parsedPcaComponents > 0 ? parsedPcaComponents : 3,
      pca_save_format: pcaSaveFormat,
    };
  }

  async function submitRun(
    url: string,
    request: ProcessFeaturesRunRequest | SegmentationRunRequest | TrackingRunRequest
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
    if (!processFeaturesOutputPath) {
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
    if (!saveHighResolutionFeatures && !savePca) {
      setRunFeedback({
        tone: "error",
        message: "Choose at least one output: Save high-resolution features and/or Save PCA.",
      });
      return;
    }

    await submitRun("/api/post-processing/process-features/run", request);
  }

  async function handleSegmentationRun() {
    if (!inputPath) {
      setRunFeedback({ tone: "error", message: "Choose an inference output folder." });
      return;
    }
    if (selectedGpuIndex === null) {
      setRunFeedback({ tone: "error", message: "Select one GPU." });
      return;
    }

    if (selectedSegmentationMode === "voronoi_otsu") {
      if (!segmentationOutputPath) {
        setRunFeedback({ tone: "error", message: "Choose a segmentation output folder." });
        return;
      }
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
        output_path: segmentationOutputPath,
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
        seed: 1337,
      });
      return;
    }

    if (!runDensityEstimation && !runProbabilityMapStage2) {
      setRunFeedback({ tone: "error", message: "Choose at least one stage: Run stage 1 and/or Run stage 2." });
      return;
    }
    if (runProbabilityMapStage2 && !segmentationOutputPath) {
      setRunFeedback({ tone: "error", message: "Choose a segmentation output folder." });
      return;
    }

    const selectedTrainingTimepoint =
      validatedSubfolderNames.length === 1 ? validatedSubfolderNames[0] ?? "" : densityEstimationTimepoint;

    const parsedFeatureBatch = Number.parseInt(probabilityMapFeatureBatch.trim(), 10);
    if (!Number.isFinite(parsedFeatureBatch) || parsedFeatureBatch < 1) {
      setRunFeedback({ tone: "error", message: "Enter a valid positive integer for Feature batch." });
      return;
    }

    const parsedKdePoints = Number.parseInt(probabilityMapKdePoints.trim(), 10);
    if (!Number.isFinite(parsedKdePoints) || parsedKdePoints < 2) {
      setRunFeedback({ tone: "error", message: "Enter a valid integer of at least 2 for Density grid size." });
      return;
    }

    const parsedKdeMaxSamples = Number.parseInt(probabilityMapKdeMaxSamples.trim(), 10);
    if (!Number.isFinite(parsedKdeMaxSamples) || parsedKdeMaxSamples < 1) {
      setRunFeedback({ tone: "error", message: "Enter a valid positive integer for Max samples per class." });
      return;
    }

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

    const parsedSeed = Number.parseInt(probabilityMapSeed.trim(), 10);
    if (!Number.isFinite(parsedSeed)) {
      setRunFeedback({ tone: "error", message: "Enter a valid integer for Random seed." });
      return;
    }

    const parsedHistSigmaBins = Number.parseFloat(probabilityMapHistSigmaBins.trim());
    if (
      probabilityMapDensityMethod === "gpu-hist" &&
      (!Number.isFinite(parsedHistSigmaBins) || parsedHistSigmaBins <= 0)
    ) {
      setRunFeedback({ tone: "error", message: "Enter a valid positive number for Histogram sigma." });
      return;
    }

    let parsedKdeBandwidth: number | null = null;
    if (probabilityMapDensityMethod === "kde" && probabilityMapKdeBandwidth.trim()) {
      parsedKdeBandwidth = Number.parseFloat(probabilityMapKdeBandwidth.trim());
      if (!Number.isFinite(parsedKdeBandwidth) || parsedKdeBandwidth <= 0) {
        setRunFeedback({ tone: "error", message: "Enter a valid positive KDE bandwidth, or leave it blank for auto." });
        return;
      }
    }

    if (runDensityEstimation) {
      if (!selectedTrainingTimepoint) {
        setRunFeedback({ tone: "error", message: "Choose one training timepoint for Stage 1." });
        return;
      }
      if (!segTifPath) {
        setRunFeedback({ tone: "error", message: "Choose an annotated FG/BG mask for Stage 1." });
        return;
      }
    } else if (runProbabilityMapStage2 && !probmapDensitiesExists) {
      setRunFeedback({
        tone: "error",
        message: "probmap_densities.npz was not found in the inference output folder root. Run Stage 1 first.",
      });
      return;
    }

    await submitRun("/api/post-processing/segmentation/run", {
      input_path: inputPath,
      output_path: runProbabilityMapStage2 ? segmentationOutputPath : null,
      gpu_index: selectedGpuIndex,
      mode: "probability_map",
      gaussian_blur_sigma: 3,
      rolling_ball_radius: 10,
      run_density_estimation: runDensityEstimation,
      run_stage_2: runProbabilityMapStage2,
      training_timepoint: runDensityEstimation ? selectedTrainingTimepoint : null,
      seg_tif: runDensityEstimation ? segTifPath : null,
      valid_mask_tif: runDensityEstimation ? validMaskTifPath : null,
      density_method: probabilityMapDensityMethod,
      feature_batch: parsedFeatureBatch,
      kde_points: parsedKdePoints,
      kde_max_samples: parsedKdeMaxSamples,
      kde_bandwidth: probabilityMapDensityMethod === "kde" ? parsedKdeBandwidth : null,
      hist_sigma_bins: probabilityMapDensityMethod === "gpu-hist" ? parsedHistSigmaBins : 1.5,
      bg_prob_threshold: parsedBgThreshold,
      fg_prob_threshold: parsedFgThreshold,
      seed: parsedSeed,
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
    if (!trackingOutputPath) {
      setRunFeedback({ tone: "error", message: "Choose an output folder." });
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
      setRunFeedback({ tone: "error", message: "Vote thresholds must be a comma-separated list of positive integers." });
      return;
    }

    await submitRun("/api/post-processing/tracking/run", {
      input_path: inputPath,
      segmentation_path: trackingSegmentationPath,
      output_path: trackingOutputPath,
      max_distance_xy: parsedMaxDistanceXy,
      max_distance_z: parsedMaxDistanceZ,
      z_distance_weight: parsedZDistanceWeight,
      min_distance_to_remove_cand: parsedMinDistanceToRemoveCand,
      vote_thresholds: normalizedVoteThresholds.join(","),
      dice_threshold: parsedDiceThreshold,
      corr_threshold: parsedCorrThreshold,
    });
  }

  return (
    <div className="preprocessPage">
      <section className="validationCard inferenceIntroCard" aria-label="Post-processing overview">
        <header className="validationHeader">
          <div>
            <h1 className="inferenceTitle">Post-processing</h1>
          </div>
        </header>
      </section>

      <section className="datasetCard" aria-label="Post-processing workflows">
        <div className="postProcessingOptionGrid">
          {WORKFLOW_OPTIONS.map((option) => {
            const active = option.value === selectedWorkflow;
            return (
              <button
                key={option.value}
                type="button"
                className={active ? "postProcessingOptionButton isActive" : "postProcessingOptionButton"}
                aria-pressed={active}
                onClick={() => setSelectedWorkflow(option.value)}
              >
                <div className="postProcessingOptionLabel">{option.label}</div>
              </button>
            );
          })}
        </div>
      </section>

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

      {inputStepVisible ? (
        <section className="datasetCard inferenceInputCard" aria-label={`${workflowLabel} folder selection`}>
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
      ) : null}

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
              <div className="inferenceFieldLabel">
                <ParameterHelpLabel
                  label="Save high-resolution features"
                  description={POST_PROCESSING_PARAMETER_HELP.saveHighResolutionFeatures}
                />
              </div>
              <label className="inferenceCheckboxLabel">
                <input
                  type="checkbox"
                  checked={saveHighResolutionFeatures}
                  onChange={(event) => setSaveHighResolutionFeatures(event.target.checked)}
                />
                <span>Enabled</span>
              </label>
              {saveHighResolutionFeatures ? (
                <>
                  <div className="inferenceInlineLabel isStrong">
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
                </>
              ) : null}
            </div>

            {saveHighResolutionFeatures ? (
              <div className="sidebarWarning">
                Saving high-resolution features writes one full 3D file per feature, typically 390 files per
                subfolder, and can consume a huge amount of disk space.
              </div>
            ) : null}

            <div className="inferenceFormRow">
              <div className="inferenceFieldLabel">
                <ParameterHelpLabel label="Save PCA" description={POST_PROCESSING_PARAMETER_HELP.savePca} />
              </div>
              <label className="inferenceCheckboxLabel">
                <input type="checkbox" checked={savePca} onChange={(event) => setSavePca(event.target.checked)} />
                <span>Enabled</span>
              </label>
              {savePca ? (
                <>
                  <div className="inferenceInlineLabel isStrong">
                    <ParameterHelpLabel label="Components" description={POST_PROCESSING_PARAMETER_HELP.components} />
                  </div>
                  <PostProcessingNumberInput value={pcaComponents} onChange={setPcaComponents} min={1} step={1} />
                  <div className="inferenceInlineLabel isStrong">
                    <ParameterHelpLabel label="Save format" description={POST_PROCESSING_PARAMETER_HELP.saveFormat} />
                  </div>
                  <select
                    className="inferenceSelect inferenceCompactSelect"
                    value={pcaSaveFormat}
                    onChange={(event) => setPcaSaveFormat(event.target.value as SaveFormat)}
                  >
                    <option value=".npy">.npy</option>
                    <option value=".tif">.tif</option>
                  </select>
                </>
              ) : null}
            </div>
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

      {parametersVisible && segmentationSelected ? (
        <section className="datasetCard inferenceFormCard" aria-label="Segmentation parameters">
          {optionsError ? <div className="sidebarError">{optionsError}</div> : null}

          <div className="inferencePathStack">
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
          </div>

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
                  ) : runProbabilityMapStage2 ? (
                    <div className={probmapDensitiesExists ? "sidebarHint" : "sidebarWarning"}>
                      {probmapDensitiesExists
                        ? `Stage 2 will load ${probmapDensitiesPath ?? "probmap_densities.npz"} from the inference output folder root.`
                        : "probmap_densities.npz was not found in the inference output folder root. Run Stage 1 first."}
                    </div>
                  ) : null}
                </div>

                <div className="postProcessingStageTitle">Stage 2: Apply probability maps</div>

                <div className="postProcessingParameterGroup">
                  <div className="postProcessingStageDescription">
                    Apply the probability maps from Stage 1 on a full movie to produce FG/BG masks, which are then
                    turned into instance segmentation masks via Connected Component Labeling.
                  </div>

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
              onClick={() => void handleSegmentationRun()}
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
            Tracking reads each inference output subfolder&apos;s <code>lr_feats.npy</code> and{" "}
            <code>volume_unnorm.tif</code>, matches them against{" "}
            <code>&lt;segmentation output folder&gt;/&lt;timepoint&gt;.tif</code>, follows the original
            overlap-and-voting tracker, and writes a single final-format <code>tracks.csv</code> into the selected
            output folder while keeping exported <code>z</code> coordinates as-is.
          </div>

          {trackingOptionalOpen ? (
            <div id="tracking-optional-parameters" className="inferenceOptionalSection">
              <div className="inferenceFormRows">
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
        selectionMode={pickerTarget === "seg_tif" || pickerTarget === "valid_mask" ? "file" : "directory"}
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
          } else {
            setValidMaskTifPath(path);
          }
          setPickerOpen(false);
        }}
      />
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
}: {
  value: string;
  onChange: (value: string) => void;
  min: number;
  step: number;
}) {
  return (
    <input
      type="number"
      className="inferenceNumberInput"
      value={value}
      onChange={(event) => onChange(event.target.value)}
      min={min}
      step={step}
    />
  );
}

function PostProcessingTextInput({
  value,
  onChange,
  placeholder,
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}) {
  return (
    <input
      type="text"
      className="inferenceNumberInput"
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
