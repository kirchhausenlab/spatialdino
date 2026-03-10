import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";

import ServerDirectoryPicker from "../components/ServerDirectoryPicker";

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

const DEFAULT_UPSAMPLE_FACTOR = "3";
const DEFAULT_ANISOTROPY = { x: "1.0", y: "1.0", z: "1.0" };

export default function InferencePage() {
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
    setFileRange({ start: "0", end: String(Math.max(0, validationResult.fileCount - 1)) });
    setCropBounds({
      xStart: "0",
      xEnd: String(Math.max(0, validationResult.shape.x - 1)),
      yStart: "0",
      yEnd: String(Math.max(0, validationResult.shape.y - 1)),
      zStart: "0",
      zEnd: String(Math.max(0, validationResult.shape.z - 1)),
    });
    setAppliedDefaultsKey(key);
  }, [appliedDefaultsKey, inputPath, validationResult]);

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
                  max={validatedDataset?.shape.x != null ? Math.max(0, validatedDataset.shape.x - 1) : undefined}
                />
                <div className="inferenceInlineLabel">X end:</div>
                <InferenceNumberInput
                  value={cropBounds.xEnd}
                  onChange={(value) => setCropBounds((current) => ({ ...current, xEnd: value }))}
                  min={0}
                  step={1}
                  max={validatedDataset?.shape.x != null ? Math.max(0, validatedDataset.shape.x - 1) : undefined}
                />
                <div className="inferenceInlineLabel">Y start:</div>
                <InferenceNumberInput
                  value={cropBounds.yStart}
                  onChange={(value) => setCropBounds((current) => ({ ...current, yStart: value }))}
                  min={0}
                  step={1}
                  max={validatedDataset?.shape.y != null ? Math.max(0, validatedDataset.shape.y - 1) : undefined}
                />
                <div className="inferenceInlineLabel">Y end:</div>
                <InferenceNumberInput
                  value={cropBounds.yEnd}
                  onChange={(value) => setCropBounds((current) => ({ ...current, yEnd: value }))}
                  min={0}
                  step={1}
                  max={validatedDataset?.shape.y != null ? Math.max(0, validatedDataset.shape.y - 1) : undefined}
                />
                <div className="inferenceInlineLabel">Z start:</div>
                <InferenceNumberInput
                  value={cropBounds.zStart}
                  onChange={(value) => setCropBounds((current) => ({ ...current, zStart: value }))}
                  min={0}
                  step={1}
                  max={validatedDataset?.shape.z != null ? Math.max(0, validatedDataset.shape.z - 1) : undefined}
                />
                <div className="inferenceInlineLabel">Z end:</div>
                <InferenceNumberInput
                  value={cropBounds.zEnd}
                  onChange={(value) => setCropBounds((current) => ({ ...current, zEnd: value }))}
                  min={0}
                  step={1}
                  max={validatedDataset?.shape.z != null ? Math.max(0, validatedDataset.shape.z - 1) : undefined}
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
                max={validatedDataset ? Math.max(0, validatedDataset.fileCount - 1) : undefined}
              />
              <div className="inferenceInlineLabel">End file:</div>
              <InferenceNumberInput
                value={fileRange.end}
                onChange={(value) => setFileRange((current) => ({ ...current, end: value }))}
                min={0}
                step={1}
                max={validatedDataset ? Math.max(0, validatedDataset.fileCount - 1) : undefined}
              />
            </div>
          </div>
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
