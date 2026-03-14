import { useEffect, useMemo, useRef, useState } from "react";

import { fetchDataOptions, submitDataDownload } from "../api/data";
import type {
  DataDownloadOverwritePromptResponse,
  DataOptionsResponse,
} from "../api/data";
import { useJobs } from "../components/JobsProvider";
import Modal from "../components/Modal";

type SelectionMode = "all" | "custom";

type RunFeedback = {
  tone: "neutral" | "success" | "error";
  message: string;
};

type OverwritePromptState = Pick<
  DataDownloadOverwritePromptResponse,
  "message" | "existingDatasetCount" | "existingDatasetNames" | "existingDatasetPaths"
>;

export default function DataPage() {
  const jobs = useJobs();
  const initializedSelectionRef = useRef(false);
  const [options, setOptions] = useState<DataOptionsResponse | null>(null);
  const [optionsLoading, setOptionsLoading] = useState(true);
  const [optionsError, setOptionsError] = useState<string | null>(null);
  const [selectionMode, setSelectionMode] = useState<SelectionMode>("all");
  const [selectedDatasetNames, setSelectedDatasetNames] = useState<string[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [feedback, setFeedback] = useState<RunFeedback | null>(null);
  const [overwritePrompt, setOverwritePrompt] = useState<OverwritePromptState | null>(null);

  useEffect(() => {
    const controller = new AbortController();

    async function loadOptions() {
      setOptionsLoading(true);
      setOptionsError(null);
      try {
        const data = await fetchDataOptions(controller.signal);
        setOptions(data);
        setSelectedDatasetNames((prev) => {
          const availableNames = data.datasets.map((dataset) => dataset.name);
          if (!initializedSelectionRef.current) {
            initializedSelectionRef.current = true;
            return availableNames;
          }
          const available = new Set(availableNames);
          return prev.filter((name) => available.has(name));
        });
      } catch (error) {
        if (controller.signal.aborted) return;
        const message = error instanceof Error ? error.message : "Unknown error";
        setOptionsError(message);
      } finally {
        if (!controller.signal.aborted) {
          setOptionsLoading(false);
        }
      }
    }

    void loadOptions();
    return () => controller.abort();
  }, []);

  const datasetNames = useMemo(() => options?.datasets.map((dataset) => dataset.name) ?? [], [options]);
  const datasetCount = datasetNames.length;
  const effectiveSelection = selectionMode === "all" ? datasetNames : selectedDatasetNames;

  useEffect(() => {
    setFeedback(null);
    setOverwritePrompt(null);
  }, [selectionMode, selectedDatasetNames]);

  function toggleDataset(name: string) {
    setSelectedDatasetNames((prev) => {
      if (prev.includes(name)) {
        return prev.filter((item) => item !== name);
      }
      const next = new Set(prev);
      next.add(name);
      return datasetNames.filter((datasetName) => next.has(datasetName));
    });
  }

  async function runDownload(existingMode?: "skip" | "overwrite") {
    setSubmitting(true);
    setFeedback(null);

    try {
      const response = await submitDataDownload(
        existingMode ? { datasets: effectiveSelection, existing_mode: existingMode } : { datasets: effectiveSelection }
      );

      if (response.submitted) {
        setOverwritePrompt(null);
        setFeedback({ tone: "success", message: response.message });
        await jobs.refresh();
        return;
      }

      if (response.valid && response.requiresOverwriteConfirmation) {
        setOverwritePrompt({
          message: response.message,
          existingDatasetCount: response.existingDatasetCount,
          existingDatasetNames: response.existingDatasetNames,
          existingDatasetPaths: response.existingDatasetPaths,
        });
        return;
      }

      setOverwritePrompt(null);
      setFeedback({
        tone: "reasonCode" in response && response.reasonCode === "nothing_to_download" ? "neutral" : "error",
        message: response.message,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      setFeedback({ tone: "error", message });
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="preprocessPage">
      <section className="validationCard inferenceIntroCard" aria-label="Public data overview">
        <div className="inferenceEyebrow">Downloads</div>
        <h1 className="inferenceTitle">Public data</h1>
        <div className="placeholderDescription dataPageDescription">
          Download the raw data used in the SpatialDINO paper examples. The downloaded files will be saved in the
          SpatialDINO directory under <code>data/raw_data</code>.
          <br />
          Visualize the data with{" "}
          <a href="https://kirchhausenlab.github.io/llsm_viewer/" target="_blank" rel="noreferrer">
            Mirante4D
          </a>
          .
        </div>
      </section>

      <section className="datasetCard" aria-label="Public data download options">
        <div className="datasetHeader">
          <div className="datasetTitleRow">
            <h2 className="datasetTitle">Download selection</h2>
            <span className="datasetSubtitle">
              {optionsLoading ? "Loading available datasets..." : `${datasetCount} dataset${datasetCount === 1 ? "" : "s"} available`}
            </span>
          </div>
        </div>

        <div className="dataCardRow">
          <div className="inferenceFieldLabel">Download root</div>
          <div className={options?.downloadRoot ? "datasetPath" : "datasetPath isEmpty"}>
            <div className="datasetPathValue">{options?.downloadRoot ?? "Loading target folder..."}</div>
          </div>
        </div>

        <div className="dataSelectionModes" role="radiogroup" aria-label="Dataset download mode">
          <label className="dataSelectionOption">
            <input
              type="radio"
              name="data-selection-mode"
              checked={selectionMode === "all"}
              onChange={() => setSelectionMode("all")}
            />
            <span>Download all data</span>
          </label>
          <label className="dataSelectionOption">
            <input
              type="radio"
              name="data-selection-mode"
              checked={selectionMode === "custom"}
              onChange={() => setSelectionMode("custom")}
            />
            <span>Choose what to download</span>
          </label>
        </div>

        {selectionMode === "custom" ? (
          <div className="dataDatasetGrid" role="group" aria-label="Available datasets">
            {datasetNames.map((name) => (
              <label key={name} className="dataDatasetToggle">
                <input
                  type="checkbox"
                  checked={selectedDatasetNames.includes(name)}
                  onChange={() => toggleDataset(name)}
                  disabled={optionsLoading || submitting}
                />
                <span>{name}</span>
              </label>
            ))}
          </div>
        ) : (
          <div className="sidebarHint">The download job will include every dataset listed in the public manifest.</div>
        )}

        {optionsError ? <ValidationMessage tone="error">{optionsError}</ValidationMessage> : null}
        {feedback ? <ValidationMessage tone={feedback.tone}>{feedback.message}</ValidationMessage> : null}
        {!optionsLoading && !optionsError && datasetCount === 0 ? (
          <ValidationMessage tone="neutral">The public data manifest is available, but it contains no datasets.</ValidationMessage>
        ) : null}

        <div className="preprocessFooterActions">
          <button
            type="button"
            className="preprocessValidateButton"
            onClick={() => void runDownload()}
            disabled={optionsLoading || submitting || !!optionsError || datasetCount === 0}
          >
            {submitting ? "Submitting..." : "Download"}
          </button>
        </div>
      </section>

      <Modal
        open={overwritePrompt !== null}
        title="Existing datasets found"
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
              className="pickerSecondaryButton"
              onClick={() => void runDownload("skip")}
              disabled={submitting}
            >
              {submitting ? "Submitting..." : "Skip existing"}
            </button>
            <button
              type="button"
              className="preprocessValidateButton"
              onClick={() => void runDownload("overwrite")}
              disabled={submitting}
            >
              {submitting ? "Submitting..." : "Overwrite existing"}
            </button>
          </>
        }
      >
        {overwritePrompt ? (
          <div className="preprocessOverwriteBody">
            <div className="preprocessOverwriteHint">{overwritePrompt.message}</div>
            <div className="sidebarHint">
              {overwritePrompt.existingDatasetCount} dataset{overwritePrompt.existingDatasetCount === 1 ? "" : "s"} already
              exist locally.
            </div>
            <ul className="preprocessOverwriteList">
              {overwritePrompt.existingDatasetNames.map((name, index) => (
                <li key={name} className="preprocessOverwriteItem">
                  {name}
                  {overwritePrompt.existingDatasetPaths[index] ? ` -> ${overwritePrompt.existingDatasetPaths[index]}` : ""}
                </li>
              ))}
            </ul>
          </div>
        ) : null}
      </Modal>
    </div>
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
