import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";

import Modal from "../components/Modal";
import ServerDirectoryPicker from "../components/ServerDirectoryPicker";
import { useJobs } from "../components/JobsProvider";
import { getClientId } from "../lib/clientId";

type PickerTarget = "input" | "output";

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
  gpu_indices: number[];
  upsample_factor: number | null;
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
  anisotropy: {
    x: number | null;
    y: number | null;
    z: number | null;
  };
  file_range: {
    start: number | null;
    end: number | null;
  };
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

type RunFeedback = {
  tone: "neutral" | "success" | "error";
  message: string;
};

type OverwritePromptState = {
  request: InferenceRunRequest;
  message: string;
  outputPath: string;
  outputEntryCount: number;
  outputEntriesPreview: string[];
};

const DEFAULT_UPSAMPLE_FACTOR = "3";
const DEFAULT_ANISOTROPY = { x: "1.0", y: "1.0", z: "1.0" };

export default function InferencePage() {
  const jobs = useJobs();
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
  const [submitting, setSubmitting] = useState(false);
  const [runFeedback, setRunFeedback] = useState<RunFeedback | null>(null);
  const [overwritePrompt, setOverwritePrompt] = useState<OverwritePromptState | null>(null);
  const [commandPreviewOpen, setCommandPreviewOpen] = useState(false);
  const [commandPreviewLoading, setCommandPreviewLoading] = useState(false);
  const [commandPreview, setCommandPreview] = useState<InferenceCommandPreviewSuccess | null>(null);
  const [commandPreviewError, setCommandPreviewError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadOptions() {
      setOptionsLoading(true);
      setOptionsError(null);
      try {
        const resp = await fetch("/api/inference/options", { headers: { Accept: "application/json" } });
        if (!resp.ok) {
          throw new Error(`Inference options failed: ${resp.status} ${resp.statusText}`);
        }
        const json = (await resp.json()) as InferenceOptionsResponse;
        if (cancelled) return;
        setAvailableGpus(json.gpus);
        setGpuError(json.gpuError?.trim() || null);
        setWeights(json.backboneWeights);
        setBackboneWeight((current) => current || json.backboneWeights[0]?.value || "");
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
    setFileRange({ start: "0", end: String(validationResult.fileCount) });
    setCropBounds({
      xStart: "0",
      xEnd: String(validationResult.shape.x),
      yStart: "0",
      yEnd: String(validationResult.shape.y),
      zStart: "0",
      zEnd: String(validationResult.shape.z),
    });
    setAppliedDefaultsKey(key);
  }, [appliedDefaultsKey, inputPath, validationResult]);

  useEffect(() => {
    setRunFeedback(null);
    setOverwritePrompt(null);
  }, [inputPath, outputPath, backboneWeight, upsampleFactor, route, precision, cropBounds, anisotropy, fileRange, selectedGpuIndices]);

  const pickerTitle = pickerTarget === "output" ? "Choose the output folder" : "Choose the input folder";
  const pickerInitialPath = pickerTarget === "output" ? outputPath : inputPath;
  const validatedDataset = validationResult?.valid ? validationResult : null;
  const parametersVisible = validatedDataset !== null;

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
      gpu_indices: selectedGpuIndices,
      upsample_factor: parseNullableNumber(upsampleFactor),
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
      setRunFeedback({ tone: "error", message: "Choose both an input folder and an output folder." });
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
      setCommandPreviewError("Choose both an input folder and an output folder.");
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

  return (
    <div className="preprocessPage">
      <section className="validationCard inferenceIntroCard" aria-label="Inference overview">
        <header className="validationHeader">
          <div>
            <div className="inferenceEyebrow">Inference</div>
            <h1 className="inferenceTitle">Configure an inference run on the server.</h1>
          </div>
        </header>
        <div className="sidebarHint">Choose and validate the input folder, then configure the inference parameters.</div>
      </section>

      <section className="datasetCard inferenceInputCard" aria-label="Input folder">
        <DirectoryFieldRow
          label="Input folder"
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
          <ValidationMessage tone="neutral">Validating input folder...</ValidationMessage>
        ) : validationError ? (
          <ValidationMessage tone="error">{validationError}</ValidationMessage>
        ) : validatedDataset ? (
          <ValidationMessage tone="success">
            {`${validatedDataset.message} ${validatedDataset.fileCount} files. Shape: x=${validatedDataset.shape.x}, y=${validatedDataset.shape.y}, z=${validatedDataset.shape.z}.`}
          </ValidationMessage>
        ) : validationResult && !validationResult.valid ? (
          <ValidationMessage tone="error">{validationResult.message}</ValidationMessage>
        ) : null}
      </section>

      {parametersVisible ? (
        <section className="datasetCard inferenceFormCard" aria-label="Inference parameters">
          {optionsError ? <div className="sidebarError">{optionsError}</div> : null}

          <div className="inferencePathStack">
            <DirectoryFieldRow label="Output folder" path={outputPath} onChoose={() => setPickerTarget("output")} />
          </div>

          <div className="inferenceFormRows">
            <div className="inferenceFormRow">
              <div className="inferenceFieldLabel">Select GPUs:</div>
              <div className="inferenceCheckboxGroup">
                {optionsLoading ? <div className="sidebarHint">Loading GPUs...</div> : null}
                {!optionsLoading && availableGpus.length === 0 && !gpuError ? (
                  <div className="sidebarHint">No NVIDIA GPUs detected.</div>
                ) : null}
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
                {gpuError ? <div className="sidebarError">{gpuError}</div> : null}
              </div>
            </div>

            <div className="inferenceFormRow">
              <div className="inferenceFieldLabel">Backbone weights:</div>
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
              <button type="button" className="pickerSecondaryButton" onClick={() => {}}>
                Download weights
              </button>
              <div className="inferenceInlineLabel isStrong">Route:</div>
              <select
                className="inferenceSelect inferenceCompactSelect"
                value={route}
                onChange={(event) => setRoute(event.target.value)}
              >
                <option value="full">Full attention (default)</option>
                <option value="streaming">Streaming attention</option>
              </select>
              <div className="inferenceInlineLabel isStrong">Precision:</div>
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
              <div className="inferenceFieldLabel">Upsample factor:</div>
              <InferenceNumberInput value={upsampleFactor} onChange={setUpsampleFactor} min={0} step="any" />
              <div className="inferenceInlineLabel isStrong">Anisotropy correction:</div>
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
              <div className="inferenceFieldLabel isStrong">Crop:</div>
              <div className="inferenceInlineGroup">
                <div className="inferenceInlineLabel">X start:</div>
                <InferenceNumberInput
                  value={cropBounds.xStart}
                  onChange={(value) => setCropBounds((current) => ({ ...current, xStart: value }))}
                  min={0}
                  step={1}
                  max={validatedDataset?.shape.x}
                />
                <div className="inferenceInlineLabel">X end:</div>
                <InferenceNumberInput
                  value={cropBounds.xEnd}
                  onChange={(value) => setCropBounds((current) => ({ ...current, xEnd: value }))}
                  min={0}
                  step={1}
                  max={validatedDataset?.shape.x}
                />
                <div className="inferenceInlineLabel">Y start:</div>
                <InferenceNumberInput
                  value={cropBounds.yStart}
                  onChange={(value) => setCropBounds((current) => ({ ...current, yStart: value }))}
                  min={0}
                  step={1}
                  max={validatedDataset?.shape.y}
                />
                <div className="inferenceInlineLabel">Y end:</div>
                <InferenceNumberInput
                  value={cropBounds.yEnd}
                  onChange={(value) => setCropBounds((current) => ({ ...current, yEnd: value }))}
                  min={0}
                  step={1}
                  max={validatedDataset?.shape.y}
                />
                <div className="inferenceInlineLabel">Z start:</div>
                <InferenceNumberInput
                  value={cropBounds.zStart}
                  onChange={(value) => setCropBounds((current) => ({ ...current, zStart: value }))}
                  min={0}
                  step={1}
                  max={validatedDataset?.shape.z}
                />
                <div className="inferenceInlineLabel">Z end:</div>
                <InferenceNumberInput
                  value={cropBounds.zEnd}
                  onChange={(value) => setCropBounds((current) => ({ ...current, zEnd: value }))}
                  min={0}
                  step={1}
                  max={validatedDataset?.shape.z}
                />
              </div>
            </div>

            <div className="inferenceFormRow">
              <div className="inferenceFieldLabel isStrong">Chosen files:</div>
              <div className="inferenceInlineLabel">Start file:</div>
              <InferenceNumberInput
                value={fileRange.start}
                onChange={(value) => setFileRange((current) => ({ ...current, start: value }))}
                min={0}
                step={1}
                max={validatedDataset?.fileCount}
              />
              <div className="inferenceInlineLabel">End file:</div>
              <InferenceNumberInput
                value={fileRange.end}
                onChange={(value) => setFileRange((current) => ({ ...current, end: value }))}
                min={0}
                step={1}
                max={validatedDataset?.fileCount}
              />
            </div>

            <div className="sidebarHint">Crop and file end values are exclusive, matching the inference script.</div>
          </div>

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

function InferenceNumberInput({
  value,
  onChange,
  min,
  step,
  disabled = false,
  max,
}: {
  value: string;
  onChange: (value: string) => void;
  min: number;
  step: number | "any";
  disabled?: boolean;
  max?: number;
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
