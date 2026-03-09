import { useMemo, useState } from "react";

import ServerDirectoryPicker from "../components/ServerDirectoryPicker";

type InputDirectory = {
  id: string;
  kind: "primary" | "secondary";
  path: string | null;
};

function createInputId() {
  const randomId = globalThis.crypto?.randomUUID?.();
  if (randomId) return randomId;
  return `input_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

export default function InferencePage() {
  const [pickerOpenForId, setPickerOpenForId] = useState<string | null>(null);
  const [inputs, setInputs] = useState<InputDirectory[]>([{ id: "primary", kind: "primary", path: null }]);

  const activeInput = useMemo(
    () => inputs.find((input) => input.id === pickerOpenForId) ?? null,
    [inputs, pickerOpenForId],
  );

  const selectedCount = inputs.filter((input) => input.path).length;
  const pickerTitle =
    activeInput?.kind === "primary" ? "Choose the primary input directory" : "Choose an additional input directory";

  return (
    <div className="preprocessPage">
      <section className="validationCard inferenceIntroCard" aria-label="Inference overview">
        <header className="validationHeader">
          <div>
            <div className="inferenceEyebrow">Inference</div>
            <h1 className="inferenceTitle">Select the server directories you want to work with.</h1>
          </div>
          <div className="preprocessMeta">
            <span className="preprocessMetaKey">Selected inputs</span>
            <span className="preprocessMetaValue">{selectedCount}</span>
          </div>
        </header>
        <div className="sidebarHint">
          This keeps the original input-selection stage from the reference GUI. The backend-powered directory picker is
          active, while the downstream processing workflow has been removed.
        </div>
      </section>

      <div className="datasetList">
        {inputs.map((input, index) => {
          const title = input.kind === "primary" ? "Primary input" : "Additional input";
          const subtitle = input.kind === "secondary" ? `#${index}` : "Required";

          return (
            <section key={input.id} className="datasetCard" aria-label={title}>
              <header className="datasetHeader">
                <div className="datasetTitleRow">
                  <h2 className="datasetTitle">{title}</h2>
                </div>
                <span className="datasetSubtitle">{subtitle}</span>
              </header>

              <div className="datasetControls">
                <button
                  type="button"
                  className="pickerPrimaryButton"
                  onClick={() => setPickerOpenForId(input.id)}
                >
                  Choose directory
                </button>

                <div className={input.path ? "datasetPath" : "datasetPath isEmpty"}>
                  <div className="datasetPathValue">{input.path ?? "No directory selected yet"}</div>
                </div>

                {input.kind === "secondary" ? (
                  <button
                    type="button"
                    className="datasetRemoveButton"
                    aria-label="Remove additional input"
                    onClick={() => {
                      setInputs((current) => current.filter((item) => item.id !== input.id));
                      setPickerOpenForId((openId) => (openId === input.id ? null : openId));
                    }}
                  >
                    <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                      <path
                        d="M9 3h6l1 2h4v2h-1l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 7H4V5h4l1-2Zm1.5 7.5v9h-2v-9h2Zm5 0v9h-2v-9h2ZM9.7 5l-.5 1h5.6l-.5-1H9.7Z"
                        fill="currentColor"
                      />
                    </svg>
                  </button>
                ) : (
                  <div className="datasetRemoveSpacer" aria-hidden="true" />
                )}
              </div>
            </section>
          );
        })}
      </div>

      <div className="preprocessFooterActions">
        <button
          type="button"
          className="pickerSecondaryButton"
          onClick={() => {
            setInputs((current) => [...current, { id: createInputId(), kind: "secondary", path: null }]);
          }}
        >
          Add input directory
        </button>
      </div>

      {selectedCount > 0 ? (
        <section className="validationCard inferenceSummaryCard" aria-label="Selected inputs">
          <div className="validationHeader">
            <div className="datasetTitle">Current selection</div>
            <div className="datasetSubtitle">{selectedCount} configured</div>
          </div>
          <div className="inferenceSummaryList">
            {inputs
              .filter((input) => input.path)
              .map((input) => (
                <div key={input.id} className="validationDatasetPath">
                  {input.path}
                </div>
              ))}
          </div>
        </section>
      ) : null}

      <ServerDirectoryPicker
        open={pickerOpenForId !== null}
        title={pickerTitle}
        initialPath={activeInput?.path ?? null}
        onClose={() => setPickerOpenForId(null)}
        onSelect={(path) => {
          if (!pickerOpenForId) return;
          setInputs((current) =>
            current.map((input) => (input.id === pickerOpenForId ? { ...input, path } : input)),
          );
          setPickerOpenForId(null);
        }}
      />
    </div>
  );
}
