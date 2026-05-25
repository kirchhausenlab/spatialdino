import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";

import Modal from "../components/Modal";
import ParameterHelpLabel from "../components/ParameterHelpLabel";
import ServerDirectoryPicker from "../components/ServerDirectoryPicker";
import { useJobs } from "../components/JobsProvider";
import { getClientId } from "../lib/clientId";

type PickerTarget = "input" | "output";
type NormalizationMode = "per_volume" | "global_auto" | "global_manual";
type BackboneModelType = "nope" | "rope" | "learned";

type GpuOption = {
  index: number;
  name: string;
};

type BackboneWeightOption = {
  label: string;
  value: string;
};

type InferenceOptionsResponse = {
  gpus: GpuOption[];
  gpuError?: string | null;
  nvidiaSmiAvailable: boolean;
  backboneWeights: BackboneWeightOption[];
};

type DatasetShape = {
  x: number;
  y: number;
  z: number;
};

type AxisNumberRequest = {
  x: number | null;
  y: number | null;
  z: number | null;
};

type AxisInputValues = {
  x: string;
  y: string;
  z: string;
};

type InferenceValidationSuccess = {
  valid: true;
  message: string;
  fileCount: number;
  shape: DatasetShape;
};

type InferenceValidationFailure = {
  valid: false;
  reasonCode: string;
  message: string;
};

type InferenceValidationResponse = InferenceValidationSuccess | InferenceValidationFailure;

type InferenceRunRequest = {
  input_path: string;
  output_path: string;
  backbone_weight: string;
  backbone_model: BackboneModelType;
  gpu_indices: number[];
  upsample_factor: AxisNumberRequest;
  route: string;
  precision: string;
  crop_bounds: {
    x_start: number | null;
    x_end: number | null;
    y_start: number | null;
    y_end: number | null;
    z_start: number | null;
    z_end: number | null;
  };
  anisotropy: AxisNumberRequest;
  file_range: {
    start: number | null;
    end: number | null;
  };
  normalization_mode: NormalizationMode;
  global_hist_min: number | null;
  global_hist_max: number | null;
  overwrite: boolean;
};

type InferenceRunSubmittedResponse = {
  submitted: true;
  jobId: string;
  message: string;
};

type InferenceRunInvalidResponse = {
  submitted: false;
  valid: false;
  reasonCode: string;
  message: string;
  requiresOverwriteConfirmation: false;
};

type InferenceRunOverwriteResponse = {
  submitted: false;
  valid: true;
  message: string;
  requiresOverwriteConfirmation: true;
  outputPath: string;
  outputEntryCount: number;
  outputEntriesPreview: string[];
};

type InferenceRunResponse = InferenceRunSubmittedResponse | InferenceRunInvalidResponse | InferenceRunOverwriteResponse;

type InferenceCommandPreviewSuccess = {
  valid: true;
  workingDirectory: string;
  command: string;
  requiresOverwriteConfirmation: boolean;
  overwriteMessage: string | null;
};

type InferenceCommandPreviewFailure = {
  valid: false;
  reasonCode: string;
  message: string;
};

type InferenceCommandPreviewResponse = InferenceCommandPreviewSuccess | InferenceCommandPreviewFailure;

type DownloadBackboneSuccessResponse = {
  downloaded: true;
  message: string;
  targetPath: string;
  backboneWeight: string;
  alreadyExisted: boolean;
};

type DownloadBackboneOverwriteResponse = {
  downloaded: false;
  requiresOverwriteConfirmation: true;
  message: string;
  targetPath: string;
  backboneWeight: string;
};

type DownloadBackboneResponse = DownloadBackboneSuccessResponse | DownloadBackboneOverwriteResponse;

type RunFeedback = {
  tone: "neutral" | "success" | "error";
  message: string;
};

type DownloadBackboneOverwritePromptState = {
  message: string;
  targetPath: string;
};

type OverwritePromptState = {
  request: InferenceRunRequest;
  message: string;
  outputPath: string;
  outputEntryCount: number;
  outputEntriesPreview: string[];
};

const DEFAULT_UPSAMPLE_FACTOR: AxisInputValues = { x: "3", y: "3", z: "3" };
const DEFAULT_ANISOTROPY: AxisInputValues = { x: "1.0", y: "1.0", z: "1.0" };
const DEFAULT_BACKBONE_MODEL: BackboneModelType = "nope";
const BACKBONE_MODEL_OPTIONS: Array<{ label: string; value: BackboneModelType }> = [
  { label: "NoPE", value: "nope" },
  { label: "RoPE", value: "rope" },
  { label: "Learned", value: "learned" },
];
const INFERENCE_OUTPUT_FOLDER_DESCRIPTION = "folder containing the outputs of a SpatialDINO run";
const INFERENCE_PARAMETER_HELP = {
  inputFolder: "Folder containing the raw data volumes to process.",
  outputFolder: INFERENCE_OUTPUT_FOLDER_DESCRIPTION,
  selectGpus: "Choose which detected GPUs will run the inference job.",
  modelWeights: "Select the pretrained checkpoint used to encode the input volumes.",
  backboneModel: "Choose the positional encoding and feed-forward settings for the selected checkpoint.",
  route: "Pick the attention implementation; streaming uses less memory on large volumes.",
  precision: "Set the numeric precision used while running inference.",
  normalization: "Controls how input intensities are scaled before the model runs.",
  upsampleFactor: "Scales the low-resolution feature grid before outputs are saved.",
  anisotropyCorrection: "Rescales axes to compensate for unequal voxel spacing.",
  crop: "Restrict inference to an inclusive subvolume in X, Y, and Z.",
  chosenFiles: "Limit inference to a contiguous range of input files.",
} as const;

export default function InferencePage() {
  const jobs = useJobs();
  const optionsRequestIdRef = useRef(0);
  const validationRequestIdRef = useRef(0);
  const [pickerTarget, setPickerTarget] = useState<PickerTarget | null>(null);
  const [inputPath, setInputPath] = useState<string | null>(null);
  const [outputPath, setOutputPath] = useState<string | null>(null);
  const [availableGpus, setAvailableGpus] = useState<GpuOption[]>([]);
  const [gpuError, setGpuError] = useState<string | null>(null);
  const [weights, setWeights] = useState<BackboneWeightOption[]>([]);
  const [optionsLoading, setOptionsLoading] = useState(true);
  const [optionsError, setOptionsError] = useState<string | null>(null);
  const [validationLoading, setValidationLoading] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [validationResult, setValidationResult] = useState<InferenceValidationResponse | null>(null);
  const [appliedDefaultsKey, setAppliedDefaultsKey] = useState<string | null>(null);
  const [selectedGpuIndices, setSelectedGpuIndices] = useState<number[]>([]);
  const [backboneWeight, setBackboneWeight] = useState("");
  const [backboneModel, setBackboneModel] = useState<BackboneModelType>(DEFAULT_BACKBONE_MODEL);
  const [upsampleFactor, setUpsampleFactor] = useState(DEFAULT_UPSAMPLE_FACTOR);
  const [route, setRoute] = useState("full");
  const [precision, setPrecision] = useState("bfloat16");
  const [cropBounds, setCropBounds] = useState({
    xStart: "0",
    xEnd: "",
    yStart: "0",
    yEnd: "",
    zStart: "0",
    zEnd: "",
  });
  const [anisotropy, setAnisotropy] = useState(DEFAULT_ANISOTROPY);
  const [fileRange, setFileRange] = useState({ start: "0", end: "" });
  const [normalizationMode, setNormalizationMode] = useState<NormalizationMode>("per_volume");
  const [globalHistMin, setGlobalHistMin] = useState("");
  const [globalHistMax, setGlobalHistMax] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [runFeedback, setRunFeedback] = useState<RunFeedback | null>(null);
  const [overwritePrompt, setOverwritePrompt] = useState<OverwritePromptState | null>(null);
  const [downloadingWeights, setDownloadingWeights] = useState(false);
  const [downloadWeightsFeedback, setDownloadWeightsFeedback] = useState<RunFeedback | null>(null);
  const [downloadWeightsOverwritePrompt, setDownloadWeightsOverwritePrompt] =
    useState<DownloadBackboneOverwritePromptState | null>(null);
  const [commandPreviewOpen, setCommandPreviewOpen] = useState(false);
  const [commandPreviewLoading, setCommandPreviewLoading] = useState(false);
  const [commandPreview, setCommandPreview] = useState<InferenceCommandPreviewSuccess | null>(null);
  const [commandPreviewError, setCommandPreviewError] = useState<string | null>(null);
  const [optionalParametersOpen, setOptionalParametersOpen] = useState(false);

  async function loadOptions(preferredBackboneWeight?: string) {
    const requestId = optionsRequestIdRef.current + 1;
    optionsRequestIdRef.current = requestId;
    setOptionsLoading(true);
    setOptionsError(null);

    try {
      const resp = await fetch("/api/inference/options", { headers: { Accept: "application/json" } });
      if (!resp.ok) {
        throw new Error(`Inference options failed: ${resp.status} ${resp.statusText}`);
      }
      const json = (await resp.json()) as InferenceOptionsResponse;
      if (requestId !== optionsRequestIdRef.current) return;

      setAvailableGpus(json.gpus);
      setGpuError(json.gpuError?.trim() || null);
      setWeights(json.backboneWeights);
      setBackboneWeight((current) => {
        if (preferredBackboneWeight && json.backboneWeights.some((weight) => weight.value === preferredBackboneWeight)) {
          return preferredBackboneWeight;
        }
        if (current && json.backboneWeights.some((weight) => weight.value === current)) {
          return current;
        }
        return json.backboneWeights[0]?.value || "";
      });
    } catch (error) {
      if (requestId !== optionsRequestIdRef.current) return;
      const message = error instanceof Error ? error.message : "Unknown error";
      setOptionsError(message);
    } finally {
      if (requestId === optionsRequestIdRef.current) {
        setOptionsLoading(false);
      }
    }
  }

  useEffect(() => {
    void loadOptions();

    return () => {
      optionsRequestIdRef.current += 1;
    };
  }, []);

  useEffect(() => {
    validationRequestIdRef.current += 1;
    setValidationLoading(false);
    setValidationResult(null);
    setValidationError(null);
  }, [inputPath]);

  useEffect(() => {
    if (!validationResult?.valid) return;

    const key = [
      inputPath ?? "",
      validationResult.fileCount,
      validationResult.shape.x,
      validationResult.shape.y,
      validationResult.shape.z,
    ].join(":");

    if (key === appliedDefaultsKey) return;

    setUpsampleFactor(DEFAULT_UPSAMPLE_FACTOR);
    setAnisotropy(DEFAULT_ANISOTROPY);
    setFileRange({ start: "0", end: formatInclusiveEndDefault(validationResult.fileCount) });
    setCropBounds({
      xStart: "0",
      xEnd: formatInclusiveEndDefault(validationResult.shape.x),
      yStart: "0",
      yEnd: formatInclusiveEndDefault(validationResult.shape.y),
      zStart: "0",
      zEnd: formatInclusiveEndDefault(validationResult.shape.z),
    });
    setAppliedDefaultsKey(key);
  }, [appliedDefaultsKey, inputPath, validationResult]);

  useEffect(() => {
    setRunFeedback(null);
    setOverwritePrompt(null);
  }, [
    inputPath,
    outputPath,
    backboneWeight,
    backboneModel,
    upsampleFactor,
    route,
    precision,
    cropBounds,
    anisotropy,
    fileRange,
    selectedGpuIndices,
    normalizationMode,
    globalHistMin,
    globalHistMax,
  ]);

  const pickerTitle = pickerTarget === "output" ? "Choose the inference output folder" : "Choose the raw data folder";
  const pickerInitialPath = pickerTarget === "output" ? outputPath : inputPath;
  const validatedDataset = validationResult?.valid ? validationResult : null;
  const parametersVisible = validatedDataset !== null;
  const maxFileIndex = validatedDataset ? Math.max(validatedDataset.fileCount - 1, 0) : undefined;
  const maxCropBounds = validatedDataset
    ? {
        x: Math.max(validatedDataset.shape.x - 1, 0),
        y: Math.max(validatedDataset.shape.y - 1, 0),
        z: Math.max(validatedDataset.shape.z - 1, 0),
      }
    : null;
  const primaryParameterMessages: ReactNode[] = [];
  const optionalParameterMessages: ReactNode[] = [];

  if (optionsError) {
    primaryParameterMessages.push(
      <ValidationMessage key="options-error" tone="error">
        {optionsError}
      </ValidationMessage>,
    );
  } else if (optionsLoading) {
    primaryParameterMessages.push(
      <div key="options-loading" className="sidebarHint">
        Loading inference options...
      </div>,
    );
  }

  if (!optionsLoading && availableGpus.length === 0 && !gpuError) {
    primaryParameterMessages.push(
      <div key="gpu-empty" className="sidebarHint">
        No NVIDIA GPUs detected.
      </div>,
    );
  }

  if (gpuError) {
    primaryParameterMessages.push(
      <div key="gpu-error" className="sidebarError">
        {gpuError}
      </div>,
    );
  }

  if (downloadWeightsFeedback) {
    primaryParameterMessages.push(
      <ValidationMessage key="download-feedback" tone={downloadWeightsFeedback.tone}>
        {downloadWeightsFeedback.message}
      </ValidationMessage>,
    );
  }

  if (route === "streaming") {
    optionalParameterMessages.push(
      <div key="route-warning" className="sidebarWarning">
        Streaming attention only works on modern GPUs.
      </div>,
    );
  }

  if (precision === "bfloat16") {
    optionalParameterMessages.push(
      <div key="precision-warning" className="sidebarWarning">
        bfloat16 only works on modern GPUs.
      </div>,
    );
    optionalParameterMessages.push(
      <div key="memory-warning" className="sidebarWarning">
        In case of GPU memory limitations, try Streaming attention or reduce the upsample factor.
      </div>,
    );
  }

  if (normalizationMode === "global_auto") {
    optionalParameterMessages.push(
      <div key="normalization-auto" className="sidebarHint">
        Auto mode runs a normalization prepass on the selected files, writes <code>norm_per_vol.txt</code> to the
        output folder, and then launches inference with those shared values.
      </div>,
    );
  }

  optionalParameterMessages.push(
    <div key="inclusive-note" className="sidebarHint">
      Crop and file end values are inclusive.
    </div>,
  );

  async function validateInputFolder() {
    if (!inputPath) return;
    const requestId = validationRequestIdRef.current + 1;
    validationRequestIdRef.current = requestId;
    setValidationLoading(true);
    setValidationError(null);

    try {
      const resp = await fetch("/api/inference/validate-input", {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ path: inputPath }),
      });

      if (requestId !== validationRequestIdRef.current) return;

      if (!resp.ok) {
        const json = await safeJson(resp);
        const detail =
          json && typeof json === "object" && "detail" in json && typeof json.detail === "string" ? json.detail : null;
        throw new Error(
          detail ? `Validation failed: ${detail}` : `Validation failed: ${resp.status} ${resp.statusText}`,
        );
      }

      const json = (await resp.json()) as InferenceValidationResponse;
      if (requestId !== validationRequestIdRef.current) return;
      setValidationResult(json);
    } catch (error) {
      if (requestId !== validationRequestIdRef.current) return;
      const message = error instanceof Error ? error.message : "Unknown error";
      setValidationError(message);
      setValidationResult(null);
    } finally {
      if (requestId === validationRequestIdRef.current) {
        setValidationLoading(false);
      }
    }
  }

  function buildRunRequest(overwrite: boolean): InferenceRunRequest | null {
    if (!inputPath || !outputPath) return null;

    return {
      input_path: inputPath,
      output_path: outputPath,
      backbone_weight: backboneWeight,
      backbone_model: backboneModel,
      gpu_indices: selectedGpuIndices,
      upsample_factor: {
        x: parseNullableNumber(upsampleFactor.x),
        y: parseNullableNumber(upsampleFactor.y),
        z: parseNullableNumber(upsampleFactor.z),
      },
      route,
      precision,
      crop_bounds: {
        x_start: parseNullableInteger(cropBounds.xStart),
        x_end: parseNullableInteger(cropBounds.xEnd),
        y_start: parseNullableInteger(cropBounds.yStart),
        y_end: parseNullableInteger(cropBounds.yEnd),
        z_start: parseNullableInteger(cropBounds.zStart),
        z_end: parseNullableInteger(cropBounds.zEnd),
      },
      anisotropy: {
        x: parseNullableNumber(anisotropy.x),
        y: parseNullableNumber(anisotropy.y),
        z: parseNullableNumber(anisotropy.z),
      },
      file_range: {
        start: parseNullableInteger(fileRange.start),
        end: parseNullableInteger(fileRange.end),
      },
      normalization_mode: normalizationMode,
      global_hist_min: normalizationMode === "global_manual" ? parseNullableNumber(globalHistMin) : null,
      global_hist_max: normalizationMode === "global_manual" ? parseNullableNumber(globalHistMax) : null,
      overwrite,
    };
  }

  async function submitInference(request: InferenceRunRequest) {
    setSubmitting(true);
    setRunFeedback(null);

    try {
      const resp = await fetch("/api/inference/run", {
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
        throw new Error(detail ? detail : `Run inference failed: ${resp.status} ${resp.statusText}`);
      }

      const json = (await resp.json()) as InferenceRunResponse;
      if (json.submitted) {
        setOverwritePrompt(null);
        setRunFeedback({ tone: "success", message: json.message });
        void jobs.refresh();
        return;
      }

      if (json.requiresOverwriteConfirmation) {
        setOverwritePrompt({
          request,
          message: json.message,
          outputPath: json.outputPath,
          outputEntryCount: json.outputEntryCount,
          outputEntriesPreview: json.outputEntriesPreview,
        });
        return;
      }

      setOverwritePrompt(null);
      setRunFeedback({ tone: "error", message: json.message });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      setRunFeedback({ tone: "error", message });
    } finally {
      setSubmitting(false);
    }
  }

  async function handleRunInference() {
    const request = buildRunRequest(false);
    if (!request) {
      setRunFeedback({ tone: "error", message: "Choose both a raw data folder and an inference output folder." });
      return;
    }
    await submitInference(request);
  }

  async function handleConfirmOverwrite() {
    if (!overwritePrompt) return;
    await submitInference({ ...overwritePrompt.request, overwrite: true });
  }

  async function handleSeeCommand() {
    setCommandPreviewOpen(true);
    setCommandPreview(null);
    setCommandPreviewError(null);

    const request = buildRunRequest(false);
    if (!request) {
      setCommandPreviewError("Choose both a raw data folder and an inference output folder.");
      return;
    }

    setCommandPreviewLoading(true);
    try {
      const resp = await fetch("/api/inference/command-preview", {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify(request),
      });

      if (!resp.ok) {
        const json = await safeJson(resp);
        const detail =
          json && typeof json === "object" && "detail" in json && typeof json.detail === "string" ? json.detail : null;
        throw new Error(detail ? detail : `Command preview failed: ${resp.status} ${resp.statusText}`);
      }

      const json = (await resp.json()) as InferenceCommandPreviewResponse;
      if (!json.valid) {
        setCommandPreviewError(json.message);
        return;
      }

      setCommandPreview(json);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      setCommandPreviewError(message);
    } finally {
      setCommandPreviewLoading(false);
    }
  }

  async function downloadBackboneWeights(overwrite: boolean) {
    setDownloadingWeights(true);
    setDownloadWeightsFeedback(null);

    try {
      const resp = await fetch("/api/inference/download-backbone", {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ overwrite }),
      });

      if (!resp.ok) {
        const json = await safeJson(resp);
        const detail =
          json && typeof json === "object" && "detail" in json && typeof json.detail === "string" ? json.detail : null;
        throw new Error(detail ? detail : `Download weights failed: ${resp.status} ${resp.statusText}`);
      }

      const json = (await resp.json()) as DownloadBackboneResponse;
      if (json.downloaded) {
        setDownloadWeightsOverwritePrompt(null);
        setDownloadWeightsFeedback({ tone: "success", message: json.message });
        await loadOptions(json.backboneWeight);
        return;
      }

      setDownloadWeightsOverwritePrompt({
        message: json.message,
        targetPath: json.targetPath,
      });
      setBackboneWeight((current) => current || json.backboneWeight);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      setDownloadWeightsFeedback({ tone: "error", message });
    } finally {
      setDownloadingWeights(false);
    }
  }

  async function handleDownloadWeights() {
    await downloadBackboneWeights(false);
  }

  async function handleConfirmDownloadWeightsOverwrite() {
    if (!downloadWeightsOverwritePrompt) return;
    await downloadBackboneWeights(true);
  }

  return (
    <div className="preprocessPage">
      <section className="validationCard inferenceIntroCard" aria-label="Inference overview">
        <header className="validationHeader">
          <div>
            <h1 className="inferenceTitle">Inference</h1>
          </div>
        </header>
      </section>

      <section className="datasetCard inferenceInputCard" aria-label="Raw data folder">
        <DirectoryFieldRow
          label={
            <ParameterHelpLabel label="Raw data folder" description={INFERENCE_PARAMETER_HELP.inputFolder} />
          }
          path={inputPath}
          onChoose={() => setPickerTarget("input")}
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
          <ValidationMessage tone="neutral">Validating raw data folder...</ValidationMessage>
        ) : validationError ? (
          <ValidationMessage tone="error">{validationError}</ValidationMessage>
        ) : validatedDataset ? (
          <ValidationMessage tone="success">
            {`Validated raw data folder. ${validatedDataset.fileCount} files. Shape: x=${validatedDataset.shape.x}, y=${validatedDataset.shape.y}, z=${validatedDataset.shape.z}.`}
          </ValidationMessage>
        ) : validationResult && !validationResult.valid ? (
          <ValidationMessage tone="error">{validationResult.message}</ValidationMessage>
        ) : null}
      </section>

      {parametersVisible ? (
        <section className="datasetCard inferenceFormCard" aria-label="Inference parameters">
          <div className="inferencePathStack">
            <DirectoryFieldRow
              label={
                <ParameterHelpLabel
                  label="Inference output folder"
                  description={INFERENCE_PARAMETER_HELP.outputFolder}
                />
              }
              path={outputPath}
              onChoose={() => setPickerTarget("output")}
            />
          </div>

          <div className="inferenceFormRows">
            <div className="inferenceFormRow">
              <div className="inferenceFieldLabel">
                <ParameterHelpLabel label="Select GPUs" description={INFERENCE_PARAMETER_HELP.selectGpus} />
              </div>
              <div className="inferenceFieldBody">
                <p className="inferenceFieldHint">
                  A single GPU processes one timepoint at a time. Using more GPUs than timepoints does
                  not bring any speed-up.
                </p>
                <div className="inferenceCheckboxGroup">
                  {availableGpus.map((gpu) => (
                    <label key={gpu.index} className="inferenceCheckboxLabel">
                      <input
                        type="checkbox"
                        checked={selectedGpuIndices.includes(gpu.index)}
                        onChange={() => {
                          setSelectedGpuIndices((current) =>
                            current.includes(gpu.index)
                              ? current.filter((value) => value !== gpu.index)
                              : [...current, gpu.index].sort((a, b) => a - b),
                          );
                        }}
                      />
                      <span>{`GPU ${gpu.index}`}</span>
                    </label>
                  ))}
                </div>
              </div>
            </div>

            <div className="inferenceFormRow">
              <div className="inferenceFieldLabel">
                <ParameterHelpLabel label="Model weights" description={INFERENCE_PARAMETER_HELP.modelWeights} />
              </div>
              <select
                className="inferenceSelect"
                value={backboneWeight}
                onChange={(event) => setBackboneWeight(event.target.value)}
                disabled={weights.length === 0}
              >
                {weights.length === 0 ? (
                  <option value="">{optionsLoading ? "Loading weights..." : "No weights found"}</option>
                ) : (
                  weights.map((weight) => (
                    <option key={weight.value} value={weight.value}>
                      {weight.label}
                    </option>
                  ))
                )}
              </select>
              <button
                type="button"
                className="pickerSecondaryButton"
                onClick={() => void handleDownloadWeights()}
                disabled={downloadingWeights}
              >
                {downloadingWeights ? "Downloading..." : "Download weights"}
              </button>
              <div className="inferenceInlineLabel isStrong">
                <ParameterHelpLabel label="Backbone type" description={INFERENCE_PARAMETER_HELP.backboneModel} />
              </div>
              <select
                className="inferenceSelect inferenceBackboneModelSelect"
                value={backboneModel}
                onChange={(event) => setBackboneModel(event.target.value as BackboneModelType)}
              >
                {BACKBONE_MODEL_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {primaryParameterMessages.length > 0 ? (
            <div className="inferenceMessages">{primaryParameterMessages}</div>
          ) : null}

          <div className="inferenceOptionalToggleRow">
            <button
              type="button"
              className="pickerPrimaryButton"
              aria-expanded={optionalParametersOpen}
              aria-controls="inference-optional-parameters"
              onClick={() => setOptionalParametersOpen((current) => !current)}
            >
              Advanced options
            </button>
          </div>

          {optionalParametersOpen ? (
            <div id="inference-optional-parameters" className="inferenceOptionalSection">
              <div className="inferenceFormRows">
                <div className="inferenceFormRow">
                  <div className="inferenceFieldLabel">
                    <ParameterHelpLabel label="Route" description={INFERENCE_PARAMETER_HELP.route} />
                  </div>
                  <select
                    className="inferenceSelect inferenceCompactSelect"
                    value={route}
                    onChange={(event) => setRoute(event.target.value)}
                  >
                    <option value="full">Full attention (default)</option>
                    <option value="streaming">Streaming attention</option>
                  </select>
                  <div className="inferenceInlineLabel isStrong">
                    <ParameterHelpLabel label="Precision" description={INFERENCE_PARAMETER_HELP.precision} />
                  </div>
                  <select
                    className="inferenceSelect inferenceCompactSelect"
                    value={precision}
                    onChange={(event) => setPrecision(event.target.value)}
                  >
                    <option value="bfloat16">bfloat16 (default)</option>
                    <option value="float16">float16</option>
                    <option value="float32">float32</option>
                  </select>
                </div>

                <div className="inferenceFormRow">
                  <div className="inferenceFieldLabel">
                    <ParameterHelpLabel label="Normalization" description={INFERENCE_PARAMETER_HELP.normalization} />
                  </div>
                  <select
                    className="inferenceSelect"
                    value={normalizationMode}
                    onChange={(event) => setNormalizationMode(event.target.value as NormalizationMode)}
                  >
                    <option value="per_volume">Per-volume (default)</option>
                    <option value="global_auto">Global, compute now</option>
                    <option value="global_manual">Global, manual values</option>
                  </select>
                  {normalizationMode === "global_manual" ? (
                    <>
                      <div className="inferenceInlineLabel">Hist min</div>
                      <InferenceNumberInput value={globalHistMin} onChange={setGlobalHistMin} min={0} step="any" />
                      <div className="inferenceInlineLabel">Hist max</div>
                      <InferenceNumberInput value={globalHistMax} onChange={setGlobalHistMax} min={0} step="any" />
                    </>
                  ) : null}
                </div>

                <div className="inferenceFormRow">
                  <div className="inferenceFieldLabel">
                    <ParameterHelpLabel label="Upsample factor" description={INFERENCE_PARAMETER_HELP.upsampleFactor} />
                  </div>
                  <div className="inferenceInlineLabel">X</div>
                  <InferenceNumberInput
                    value={upsampleFactor.x}
                    onChange={(value) => setUpsampleFactor((current) => ({ ...current, x: value }))}
                    min={1}
                    step="any"
                    ariaLabel="Upsample factor X"
                  />
                  <div className="inferenceInlineLabel">Y</div>
                  <InferenceNumberInput
                    value={upsampleFactor.y}
                    onChange={(value) => setUpsampleFactor((current) => ({ ...current, y: value }))}
                    min={1}
                    step="any"
                    ariaLabel="Upsample factor Y"
                  />
                  <div className="inferenceInlineLabel">Z</div>
                  <InferenceNumberInput
                    value={upsampleFactor.z}
                    onChange={(value) => setUpsampleFactor((current) => ({ ...current, z: value }))}
                    min={1}
                    step="any"
                    ariaLabel="Upsample factor Z"
                  />
                </div>

                <div className="inferenceFormRow">
                  <div className="inferenceFieldLabel">
                    <ParameterHelpLabel
                      label="Anisotropy correction"
                      description={INFERENCE_PARAMETER_HELP.anisotropyCorrection}
                    />
                  </div>
                  <div className="inferenceInlineLabel">X</div>
                  <InferenceNumberInput
                    value={anisotropy.x}
                    onChange={(value) => setAnisotropy((current) => ({ ...current, x: value }))}
                    min={0}
                    step="any"
                  />
                  <div className="inferenceInlineLabel">Y</div>
                  <InferenceNumberInput
                    value={anisotropy.y}
                    onChange={(value) => setAnisotropy((current) => ({ ...current, y: value }))}
                    min={0}
                    step="any"
                  />
                  <div className="inferenceInlineLabel">Z</div>
                  <InferenceNumberInput
                    value={anisotropy.z}
                    onChange={(value) => setAnisotropy((current) => ({ ...current, z: value }))}
                    min={0}
                    step="any"
                  />
                </div>

                <div className="inferenceFormRow">
                  <div className="inferenceFieldLabel isStrong">
                    <ParameterHelpLabel label="Crop" description={INFERENCE_PARAMETER_HELP.crop} />
                  </div>
                  <div className="inferenceInlineGroup">
                    <div className="inferenceAxisGroup">
                      <div className="inferenceInlineLabel">X</div>
                      <InferenceNumberInput
                        value={cropBounds.xStart}
                        onChange={(value) => setCropBounds((current) => ({ ...current, xStart: value }))}
                        min={0}
                        step={1}
                        max={maxCropBounds?.x}
                        placeholder="start"
                        ariaLabel="Crop X start"
                      />
                      <InferenceNumberInput
                        value={cropBounds.xEnd}
                        onChange={(value) => setCropBounds((current) => ({ ...current, xEnd: value }))}
                        min={0}
                        step={1}
                        max={maxCropBounds?.x}
                        placeholder="end"
                        ariaLabel="Crop X end"
                      />
                    </div>
                    <div className="inferenceAxisGroup">
                      <div className="inferenceInlineLabel">Y</div>
                      <InferenceNumberInput
                        value={cropBounds.yStart}
                        onChange={(value) => setCropBounds((current) => ({ ...current, yStart: value }))}
                        min={0}
                        step={1}
                        max={maxCropBounds?.y}
                        placeholder="start"
                        ariaLabel="Crop Y start"
                      />
                      <InferenceNumberInput
                        value={cropBounds.yEnd}
                        onChange={(value) => setCropBounds((current) => ({ ...current, yEnd: value }))}
                        min={0}
                        step={1}
                        max={maxCropBounds?.y}
                        placeholder="end"
                        ariaLabel="Crop Y end"
                      />
                    </div>
                    <div className="inferenceAxisGroup">
                      <div className="inferenceInlineLabel">Z</div>
                      <InferenceNumberInput
                        value={cropBounds.zStart}
                        onChange={(value) => setCropBounds((current) => ({ ...current, zStart: value }))}
                        min={0}
                        step={1}
                        max={maxCropBounds?.z}
                        placeholder="start"
                        ariaLabel="Crop Z start"
                      />
                      <InferenceNumberInput
                        value={cropBounds.zEnd}
                        onChange={(value) => setCropBounds((current) => ({ ...current, zEnd: value }))}
                        min={0}
                        step={1}
                        max={maxCropBounds?.z}
                        placeholder="end"
                        ariaLabel="Crop Z end"
                      />
                    </div>
                  </div>
                </div>

                <div className="inferenceFormRow">
                  <div className="inferenceFieldLabel isStrong">
                    <ParameterHelpLabel label="Chosen files" description={INFERENCE_PARAMETER_HELP.chosenFiles} />
                  </div>
                  <div className="inferenceInlineLabel">Start file:</div>
                  <InferenceNumberInput
                    value={fileRange.start}
                    onChange={(value) => setFileRange((current) => ({ ...current, start: value }))}
                    min={0}
                    step={1}
                    max={maxFileIndex}
                    ariaLabel="Start file"
                  />
                  <div className="inferenceInlineLabel">End file:</div>
                  <InferenceNumberInput
                    value={fileRange.end}
                    onChange={(value) => setFileRange((current) => ({ ...current, end: value }))}
                    min={0}
                    step={1}
                    max={maxFileIndex}
                    ariaLabel="End file"
                  />
                </div>
              </div>

              {optionalParameterMessages.length > 0 ? (
                <div className="inferenceMessages">{optionalParameterMessages}</div>
              ) : null}
            </div>
          ) : null}

          <div className="validationActions">
            <button
              type="button"
              className="preprocessValidateButton"
              disabled={submitting}
              onClick={() => void handleRunInference()}
            >
              {submitting ? "Submitting..." : "Run inference"}
            </button>
            <button
              type="button"
              className="pickerSecondaryButton"
              disabled={submitting || commandPreviewLoading}
              onClick={() => void handleSeeCommand()}
            >
              {commandPreviewLoading ? "Loading..." : "See command"}
            </button>
          </div>

          {runFeedback ? <ValidationMessage tone={runFeedback.tone}>{runFeedback.message}</ValidationMessage> : null}
        </section>
      ) : null}

      <ServerDirectoryPicker
        open={pickerTarget !== null}
        title={pickerTitle}
        initialPath={pickerInitialPath}
        onClose={() => setPickerTarget(null)}
        onSelect={(path) => {
          if (pickerTarget === "input") {
            setInputPath(path);
          } else if (pickerTarget === "output") {
            setOutputPath(path);
          }
          setPickerTarget(null);
        }}
      />

      <Modal
        open={commandPreviewOpen}
        title="Inference command"
        onClose={() => {
          if (commandPreviewLoading) return;
          setCommandPreviewOpen(false);
        }}
        footer={
          <button
            type="button"
            className="pickerSecondaryButton"
            onClick={() => setCommandPreviewOpen(false)}
            disabled={commandPreviewLoading}
          >
            Close
          </button>
        }
      >
        {commandPreviewLoading ? (
          <div className="sidebarHint">Preparing the CLI command...</div>
        ) : commandPreviewError ? (
          <ValidationMessage tone="error">{commandPreviewError}</ValidationMessage>
        ) : commandPreview ? (
          <>
            <div className="sidebarHint">This is the CLI command the server would launch from the repo root.</div>
            {commandPreview.requiresOverwriteConfirmation && commandPreview.overwriteMessage ? (
              <div className="sidebarWarning">{commandPreview.overwriteMessage}</div>
            ) : null}
            <div className="inferenceCommandMeta">
              <div className="inferenceStrongText">Working directory</div>
              <div className="datasetPath">
                <div className="datasetPathValue">{commandPreview.workingDirectory}</div>
              </div>
            </div>
            <pre className="inferenceCommandPreview">
              <code>{commandPreview.command}</code>
            </pre>
          </>
        ) : null}
      </Modal>

      <Modal
        open={downloadWeightsOverwritePrompt !== null}
        title="Overwrite weights file?"
        onClose={() => {
          if (downloadingWeights) return;
          setDownloadWeightsOverwritePrompt(null);
        }}
        footer={
          <>
            <button
              type="button"
              className="pickerSecondaryButton"
              onClick={() => setDownloadWeightsOverwritePrompt(null)}
              disabled={downloadingWeights}
            >
              Cancel
            </button>
            <button
              type="button"
              className="preprocessValidateButton"
              onClick={() => void handleConfirmDownloadWeightsOverwrite()}
              disabled={downloadingWeights}
            >
              {downloadingWeights ? "Downloading..." : "Overwrite and download"}
            </button>
          </>
        }
      >
        {downloadWeightsOverwritePrompt ? (
          <div className="preprocessOverwriteBody">
            <div className="preprocessOverwriteHint">{downloadWeightsOverwritePrompt.message}</div>
            <div className="datasetPath">
              <div className="datasetPathValue">{downloadWeightsOverwritePrompt.targetPath}</div>
            </div>
            <div className="sidebarHint">
              Overwriting will replace the current `backbone.pth` in the weights folder.
            </div>
          </div>
        ) : null}
      </Modal>

      <Modal
        open={overwritePrompt !== null}
        title="Overwrite output folder?"
        onClose={() => {
          if (submitting) return;
          setOverwritePrompt(null);
        }}
        footer={
          <>
            <button
              type="button"
              className="pickerSecondaryButton"
              onClick={() => setOverwritePrompt(null)}
              disabled={submitting}
            >
              Cancel
            </button>
            <button
              type="button"
              className="preprocessValidateButton"
              onClick={() => void handleConfirmOverwrite()}
              disabled={submitting}
            >
              {submitting ? "Submitting..." : "Overwrite and run"}
            </button>
          </>
        }
      >
        {overwritePrompt ? (
          <div className="preprocessOverwriteBody">
            <div className="preprocessOverwriteHint">{overwritePrompt.message}</div>
            <div className="datasetPath">
              <div className="datasetPathValue">{overwritePrompt.outputPath}</div>
            </div>
            <div className="sidebarHint">
              {`${overwritePrompt.outputEntryCount} existing item${overwritePrompt.outputEntryCount === 1 ? "" : "s"} will be removed before the job starts.`}
            </div>
            {overwritePrompt.outputEntriesPreview.length > 0 ? (
              <ul className="preprocessOverwriteList">
                {overwritePrompt.outputEntriesPreview.map((entry) => (
                  <li key={entry} className="preprocessOverwriteItem">
                    {entry}
                  </li>
                ))}
              </ul>
            ) : null}
            {overwritePrompt.outputEntryCount > overwritePrompt.outputEntriesPreview.length ? (
              <div className="sidebarHint">
                {`And ${overwritePrompt.outputEntryCount - overwritePrompt.outputEntriesPreview.length} more item(s).`}
              </div>
            ) : null}
          </div>
        ) : null}
      </Modal>
    </div>
  );
}

function DirectoryFieldRow({
  label,
  path,
  onChoose,
  action,
}: {
  label: ReactNode;
  path: string | null;
  onChoose: () => void;
  action?: ReactNode;
}) {
  return (
    <div className="inferencePathRow">
      <div className="inferencePathLabel">{label}</div>
      <button type="button" className="pickerPrimaryButton" onClick={onChoose}>
        Choose directory
      </button>
      <div className={path ? "datasetPath" : "datasetPath isEmpty"}>
        <div className="datasetPathValue">{path ?? "No directory selected yet"}</div>
      </div>
      {action ? <div className="inferencePathAction">{action}</div> : null}
    </div>
  );
}

function InferenceNumberInput({
  value,
  onChange,
  min,
  step,
  disabled = false,
  max,
  placeholder,
  ariaLabel,
}: {
  value: string;
  onChange: (value: string) => void;
  min: number;
  step: number | "any";
  disabled?: boolean;
  max?: number;
  placeholder?: string;
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
      disabled={disabled}
      max={max}
      placeholder={placeholder}
      aria-label={ariaLabel}
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

async function safeJson(resp: Response): Promise<any> {
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

function parseNullableNumber(value: string): number | null {
  const trimmed = value.trim();
  if (!trimmed) return null;
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatInclusiveEndDefault(size: number): string {
  return size > 0 ? String(size - 1) : "";
}
