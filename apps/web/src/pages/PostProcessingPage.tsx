import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";

import { useJobs } from "../components/JobsProvider";
import ServerDirectoryPicker from "../components/ServerDirectoryPicker";
import { getClientId } from "../lib/clientId";

type WorkflowOption = "process_features" | "segmentation" | "tracking";
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
};

type PostProcessingValidationFailure = {
  valid: false;
  reasonCode: string;
  message: string;
};

type PostProcessingValidationResult = PostProcessingValidationSuccess | PostProcessingValidationFailure;

type ProcessFeaturesRunRequest = {
  input_path: string;
  gpu_index: number | null;
  save_high_resolution_features: boolean;
  high_resolution_save_format: SaveFormat;
  save_pca: boolean;
  pca_components: number;
  pca_save_format: SaveFormat;
};

type SegmentationRunRequest = {
  input_path: string;
  gpu_index: number | null;
  enable_voronoi_otsu: boolean;
  gaussian_blur_sigma: number;
  rolling_ball_radius: number;
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

function getWorkflowLabel(workflow: WorkflowOption | null): string {
  if (!workflow) return "Post-processing";
  return WORKFLOW_OPTIONS.find((option) => option.value === workflow)?.label ?? "Post-processing";
}

export default function PostProcessingPage() {
  const jobs = useJobs();
  const validationRequestIdRef = useRef(0);

  const [selectedWorkflow, setSelectedWorkflow] = useState<WorkflowOption | null>(null);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [inputPath, setInputPath] = useState<string | null>(null);
  const [validationLoading, setValidationLoading] = useState(false);
  const [validationResult, setValidationResult] = useState<PostProcessingValidationResult | null>(null);
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
  const [enableVoronoiOtsu, setEnableVoronoiOtsu] = useState(true);
  const [gaussianBlurSigma, setGaussianBlurSigma] = useState("3");
  const [rollingBallRadius, setRollingBallRadius] = useState("10");
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
  }, [inputPath]);

  useEffect(() => {
    setRunFeedback(null);
  }, [
    selectedWorkflow,
    inputPath,
    selectedGpuIndex,
    saveHighResolutionFeatures,
    highResolutionSaveFormat,
    savePca,
    pcaComponents,
    pcaSaveFormat,
    enableVoronoiOtsu,
    gaussianBlurSigma,
    rollingBallRadius,
  ]);

  const workflowLabel = getWorkflowLabel(selectedWorkflow);
  const inputStepVisible = selectedWorkflow !== null;
  const processFeaturesSelected = selectedWorkflow === "process_features";
  const segmentationSelected = selectedWorkflow === "segmentation";
  const trackingSelected = selectedWorkflow === "tracking";
  const parametersVisible = validationResult?.valid === true;

  async function validateInputFolder() {
    if (!inputPath) return;

    const requestId = validationRequestIdRef.current + 1;
    validationRequestIdRef.current = requestId;
    setValidationLoading(true);
    setValidationResult(null);

    try {
      const resp = await fetch("/api/post-processing/process-features/validate-input", {
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

  function buildProcessFeaturesRunRequest(): ProcessFeaturesRunRequest | null {
    if (!inputPath) return null;

    const parsedPcaComponents = Number.parseInt(pcaComponents.trim(), 10);
    if (savePca && (!Number.isFinite(parsedPcaComponents) || parsedPcaComponents < 1)) {
      return null;
    }

    return {
      input_path: inputPath,
      gpu_index: selectedGpuIndex,
      save_high_resolution_features: saveHighResolutionFeatures,
      high_resolution_save_format: highResolutionSaveFormat,
      save_pca: savePca,
      pca_components: Number.isFinite(parsedPcaComponents) && parsedPcaComponents > 0 ? parsedPcaComponents : 3,
      pca_save_format: pcaSaveFormat,
    };
  }

  async function submitRun(url: string, request: ProcessFeaturesRunRequest | SegmentationRunRequest) {
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
      setRunFeedback({ tone: "error", message: "Choose an input folder." });
      return;
    }
    if (selectedGpuIndex === null) {
      setRunFeedback({ tone: "error", message: "Select one GPU." });
      return;
    }
    if (!enableVoronoiOtsu) {
      setRunFeedback({ tone: "error", message: "Enable Voronoi-Otsu segmentation." });
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
      gpu_index: selectedGpuIndex,
      enable_voronoi_otsu: enableVoronoiOtsu,
      gaussian_blur_sigma: parsedGaussianBlurSigma,
      rolling_ball_radius: parsedRollingBallRadius,
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

      {inputStepVisible ? (
        <section className="datasetCard inferenceInputCard" aria-label={`${workflowLabel} input folder`}>
          <DirectoryFieldRow
            label="Input folder"
            path={inputPath}
            onChoose={() => setPickerOpen(true)}
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
            <ValidationMessage tone="neutral">Validating input folder...</ValidationMessage>
          ) : validationResult ? (
            <ValidationMessage tone={validationResult.valid ? "success" : "error"}>{validationResult.message}</ValidationMessage>
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
            />

            <div className="inferenceFormRow">
              <div className="inferenceFieldLabel">Save high-resolution features:</div>
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
                  <div className="inferenceInlineLabel isStrong">Save format:</div>
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
              <div className="inferenceFieldLabel">Save PCA:</div>
              <label className="inferenceCheckboxLabel">
                <input type="checkbox" checked={savePca} onChange={(event) => setSavePca(event.target.checked)} />
                <span>Enabled</span>
              </label>
              {savePca ? (
                <>
                  <div className="inferenceInlineLabel isStrong">Components:</div>
                  <PostProcessingNumberInput value={pcaComponents} onChange={setPcaComponents} min={1} step={1} />
                  <div className="inferenceInlineLabel isStrong">Save format:</div>
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

          <div className="inferenceFormRows">
            <GpuSelectionRow
              optionsLoading={optionsLoading}
              availableGpus={availableGpus}
              gpuError={gpuError}
              selectedGpuIndex={selectedGpuIndex}
              onSelectGpu={setSelectedGpuIndex}
            />

            <div className="inferenceFormRow">
              <div className="inferenceFieldLabel">Voronoi-Otsu segmentation:</div>
              <label className="inferenceCheckboxLabel">
                <input
                  type="checkbox"
                  checked={enableVoronoiOtsu}
                  onChange={(event) => setEnableVoronoiOtsu(event.target.checked)}
                />
                <span>Enabled</span>
              </label>
              {enableVoronoiOtsu ? (
                <>
                  <div className="inferenceInlineLabel isStrong">Gaussian blur sigma:</div>
                  <PostProcessingNumberInput
                    value={gaussianBlurSigma}
                    onChange={setGaussianBlurSigma}
                    min={0}
                    step={1}
                  />
                  <div className="inferenceInlineLabel isStrong">Rolling ball radius:</div>
                  <PostProcessingNumberInput
                    value={rollingBallRadius}
                    onChange={setRollingBallRadius}
                    min={0}
                    step={0.1}
                  />
                </>
              ) : null}
            </div>
          </div>

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
        <section className="datasetCard postProcessingPlaceholderCard" aria-label="Tracking post-processing">
          <div className="postProcessingPlaceholderTitle">Tracking</div>
        </section>
      ) : null}

      <ServerDirectoryPicker
        open={pickerOpen}
        title="Choose the input folder"
        initialPath={inputPath}
        onClose={() => setPickerOpen(false)}
        onSelect={(path) => {
          setInputPath(path);
          setPickerOpen(false);
        }}
      />
    </div>
  );
}

function DirectoryFieldRow({
  label,
  path,
  onChoose,
  action,
}: {
  label: string;
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

function GpuSelectionRow({
  optionsLoading,
  availableGpus,
  gpuError,
  selectedGpuIndex,
  onSelectGpu,
}: {
  optionsLoading: boolean;
  availableGpus: GpuOption[];
  gpuError: string | null;
  selectedGpuIndex: number | null;
  onSelectGpu: (gpuIndex: number | null) => void;
}) {
  return (
    <div className="inferenceFormRow">
      <div className="inferenceFieldLabel">Select GPU:</div>
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
