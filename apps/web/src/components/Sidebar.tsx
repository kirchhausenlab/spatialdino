import { useCallback, useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";

import { getJobLog } from "../api/jobs";
import type { JobLogDetails, JobSummary } from "../api/jobs";
import Modal from "./Modal";
import { useJobs } from "./JobsProvider";

function getServerHostnameFromDom(): string | null {
  const meta = document.querySelector<HTMLMetaElement>('meta[name="spatialdino-server-hostname"]');
  if (!meta) return null;
  const value = meta.content?.trim();
  if (!value || value === "__SPATIALDINO_SERVER_HOSTNAME__") return null;
  return value;
}

function getSessionLabelFromDom(): string | null {
  const meta = document.querySelector<HTMLMetaElement>('meta[name="spatialdino-session-label"]');
  if (!meta) return null;
  const value = meta.content?.trim();
  if (!value || value === "__SPATIALDINO_SESSION_LABEL__") return null;
  return value;
}

type CpuStatus = {
  totalCores: number;
  activeCores: number;
  averageUtilizationPct: number;
  sampleWindowMs: number;
};

type GpuInfo = {
  index: number;
  name: string;
  memoryUsedMiB: number;
  memoryTotalMiB: number;
};

type GpuStatus = {
  nvidiaSmiAvailable: boolean;
  gpus: GpuInfo[];
  error?: string;
};

export default function Sidebar() {
  const serverHostname = getServerHostnameFromDom();
  const [sessionLabel, setSessionLabel] = useState<string | null>(() => getSessionLabelFromDom());
  const didAutoRefresh = useRef(false);
  const jobs = useJobs();

  const [cpu, setCpu] = useState<{
    loading: boolean;
    error: string | null;
    data: CpuStatus | null;
    updatedAt: Date | null;
  }>({ loading: false, error: null, data: null, updatedAt: null });

  const [gpus, setGpus] = useState<{
    loading: boolean;
    error: string | null;
    data: GpuStatus | null;
    updatedAt: Date | null;
  }>({ loading: false, error: null, data: null, updatedAt: null });
  const [jobLog, setJobLog] = useState<{
    openJobId: string | null;
    loading: boolean;
    error: string | null;
    data: JobLogDetails | null;
  }>({ openJobId: null, loading: false, error: null, data: null });
  const jobLogAbortRef = useRef<AbortController | null>(null);

  const refreshCpu = useCallback(async () => {
    setCpu((prev) => ({ ...prev, loading: true, error: null }));
    try {
      const resp = await fetch("/api/status/cpu", { headers: { Accept: "application/json" } });
      if (!resp.ok) {
        throw new Error(`CPU status failed: ${resp.status} ${resp.statusText}`);
      }
      const json = (await resp.json()) as CpuStatus;
      setCpu({ loading: false, error: null, data: json, updatedAt: new Date() });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      setCpu((prev) => ({ ...prev, loading: false, error: message }));
    }
  }, []);

  const refreshGpus = useCallback(async () => {
    setGpus((prev) => ({ ...prev, loading: true, error: null }));
    try {
      const resp = await fetch("/api/status/gpus", { headers: { Accept: "application/json" } });
      if (!resp.ok) {
        throw new Error(`GPU status failed: ${resp.status} ${resp.statusText}`);
      }
      const json = (await resp.json()) as GpuStatus;
      setGpus({ loading: false, error: null, data: json, updatedAt: new Date() });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      setGpus((prev) => ({ ...prev, loading: false, error: message }));
    }
  }, []);

  useEffect(() => {
    if (didAutoRefresh.current) return;
    didAutoRefresh.current = true;
    void refreshCpu();
    void refreshGpus();
  }, [refreshCpu, refreshGpus]);

  useEffect(() => {
    if (sessionLabel) return;

    let cancelled = false;

    async function loadSessionLabel() {
      try {
        const resp = await fetch("/api/session", { headers: { Accept: "application/json" } });
        if (!resp.ok) return;
        const json = (await resp.json()) as { sessionLabel?: string };
        const next = json.sessionLabel?.trim();
        if (!cancelled && next) {
          setSessionLabel(next);
        }
      } catch {
        // Session labeling is informational only.
      }
    }

    void loadSessionLabel();

    return () => {
      cancelled = true;
    };
  }, [sessionLabel]);

  const loadJobLog = useCallback(async (jobId: string, options?: { keepData?: boolean }) => {
    jobLogAbortRef.current?.abort();
    const controller = new AbortController();
    jobLogAbortRef.current = controller;
    setJobLog((prev) => ({
      openJobId: jobId,
      loading: true,
      error: null,
      data: options?.keepData === false || prev.openJobId !== jobId ? null : prev.data,
    }));

    try {
      const data = await getJobLog(jobId, { signal: controller.signal, tailLines: 200 });
      if (controller.signal.aborted) return;
      setJobLog({ openJobId: jobId, loading: false, error: null, data });
    } catch (error) {
      if (controller.signal.aborted) return;
      const message = error instanceof Error ? error.message : "Unknown error";
      setJobLog((prev) => ({
        openJobId: jobId,
        loading: false,
        error: message,
        data: prev.openJobId === jobId ? prev.data : null,
      }));
    }
  }, []);

  const closeJobLog = useCallback(() => {
    jobLogAbortRef.current?.abort();
    setJobLog({ openJobId: null, loading: false, error: null, data: null });
  }, []);

  useEffect(() => {
    return () => {
      jobLogAbortRef.current?.abort();
    };
  }, []);

  const selectedJob = jobLog.openJobId ? jobs.jobs.find((job) => job.jobId === jobLog.openJobId) ?? null : null;
  const selectedJobTitle =
    selectedJob?.label?.trim() ||
    (jobLog.data ? formatJobTitle(jobLog.data.type) : jobLog.openJobId ? "Job details" : "Job");

  useEffect(() => {
    if (!jobLog.openJobId || !selectedJob) return;
    if (selectedJob.status !== "running" || jobLog.loading) return;
    const timer = window.setTimeout(() => {
      void loadJobLog(jobLog.openJobId as string);
    }, 1000);
    return () => window.clearTimeout(timer);
  }, [jobLog.loading, jobLog.openJobId, loadJobLog, selectedJob]);

  return (
    <aside className="appSidebar" aria-label="Sidebar">
      <div className="sidebarInner">
        <div className="sidebarHeader">
          <div className="sidebarHeaderTitle">Session</div>
          {sessionLabel ? <div className="sidebarHeaderMeta">{sessionLabel}</div> : null}
        </div>

        <div className="sidebarRow">
          <span className="sidebarKey">Server</span>
          <span className={serverHostname ? "sidebarValue" : "sidebarValue isEmpty"}>
            {serverHostname ?? "Unknown"}
          </span>
        </div>

        <SidebarSection
          title="CPU"
          right={
            <SidebarActionButton onClick={refreshCpu} disabled={cpu.loading}>
              {cpu.loading ? "Refreshing..." : "Refresh"}
            </SidebarActionButton>
          }
        >
          {cpu.error ? <div className="sidebarError">{cpu.error}</div> : null}
          {!cpu.data && !cpu.error ? <div className="sidebarHint">Refreshing CPU activity.</div> : null}
          {cpu.data ? (
            <>
              <div className="sidebarRow">
                <span className="sidebarKey">Active cores</span>
                <span className="sidebarValue">
                  {cpu.data.activeCores}/{cpu.data.totalCores}
                </span>
              </div>
              <div className="sidebarRow">
                <span className="sidebarKey">Average utilization</span>
                <span className="sidebarValue">{cpu.data.averageUtilizationPct}%</span>
              </div>
              <div className="sidebarMeta">
                Sampled {cpu.data.sampleWindowMs}ms
                {cpu.updatedAt ? ` · Updated ${formatTime(cpu.updatedAt)}` : ""}
              </div>
            </>
          ) : null}
        </SidebarSection>

        <SidebarSection
          title="GPUs"
          right={
            <SidebarActionButton onClick={refreshGpus} disabled={gpus.loading}>
              {gpus.loading ? "Refreshing..." : "Refresh"}
            </SidebarActionButton>
          }
        >
          {gpus.error ? <div className="sidebarError">{gpus.error}</div> : null}
          {gpus.data?.error ? <div className="sidebarError">{gpus.data.error}</div> : null}
          {!gpus.data && !gpus.error ? (
            <div className="sidebarHint">Refreshing NVIDIA GPU memory usage.</div>
          ) : null}
          {gpus.data ? (
            <>
              {gpus.data.gpus.length === 0 ? (
                <div className="sidebarHint">
                  {gpus.data.nvidiaSmiAvailable ? "Server without NVIDIA GPUs" : "nvidia-smi not available"}
                </div>
              ) : (
                <div className="sidebarList">
                  {gpus.data.gpus.map((gpu) => (
                    <div key={gpu.index} className="sidebarRow">
                      <span className="sidebarKey">GPU {gpu.index}</span>
                      <span className="sidebarValue">
                        {gpu.memoryUsedMiB}/{gpu.memoryTotalMiB} MiB ({pct(gpu.memoryUsedMiB, gpu.memoryTotalMiB)}%)
                      </span>
                    </div>
                  ))}
                </div>
              )}
              {gpus.updatedAt ? <div className="sidebarMeta">Updated {formatTime(gpus.updatedAt)}</div> : null}
            </>
          ) : null}
        </SidebarSection>

        <SidebarSection
          title="Jobs"
          right={
            <>
              <SidebarActionButton onClick={() => void jobs.refresh()} disabled={jobs.loading}>
                Refresh
              </SidebarActionButton>
              <SidebarActionButton
                onClick={jobs.clearFinished}
                disabled={!jobs.jobs.some((job) => job.status !== "running")}
              >
                Clear
              </SidebarActionButton>
            </>
          }
        >
          {jobs.error ? <div className="sidebarError">{jobs.error}</div> : null}
          {jobs.jobs.length === 0 ? <div className="sidebarHint">No jobs submitted.</div> : null}
          {jobs.jobs.length > 0 ? (
            <div className="sidebarJobList">
              {jobs.jobs.map((job) => {
                const pctValue = job.total > 0 ? Math.round((job.processed / job.total) * 100) : 0;
                const doneText = job.total > 0 ? `${job.processed}/${job.total}` : `${job.processed}`;
                const now = Date.now();
                const elapsedMs = (job.finishedAtMs ?? now) - job.createdAtMs;
                const elapsedText = formatDurationMs(elapsedMs);
                const remainingText = estimateRemaining(job.status, elapsedMs, job.processed, job.total);
                const statusText = `${job.status}${job.current ? ` · ${job.current}` : ""}`;
                const title = job.label?.trim() || formatJobTitle(job.type);
                const failureSummary = formatFailureSummary(job);
                const canInspect = job.type === "inference" || job.status === "failed" || Boolean(job.logAvailable);
                const detailsButtonLabel =
                  jobLog.openJobId === job.jobId && jobLog.loading
                    ? "Loading..."
                    : job.status === "running"
                      ? "Live log"
                      : "Details";
                const datasetLines = (job.datasets ?? [])
                  .map((dataset) =>
                    job.saveDir ? `${dataset.source_dir} → ${job.saveDir}/${dataset.save_to}` : dataset.source_dir
                  )
                  .join("\n");

                const tooltipLines = [
                  title,
                  `Status: ${statusText}`,
                  `Progress: ${doneText}${job.total > 0 ? ` (${pctValue}%)` : ""}`,
                  typeof job.exitCode === "number" ? `Exit code: ${job.exitCode}` : null,
                  job.error ? `Error: ${job.error}` : null,
                  job.logPath ? `Log: ${job.logPath}` : null,
                  job.roi
                    ? `ROI: x=${job.roi.x0}..${job.roi.x1}, y=${job.roi.y0}..${job.roi.y1}, z=${job.roi.z0}..${job.roi.z1}`
                    : null,
                  typeof job.addedPadding === "number" && job.addedPadding > 0
                    ? `Padding: ${job.addedPadding}${job.invertLut ? " (inverted cval)" : ""}`
                    : null,
                  typeof job.invertLut === "boolean" ? `Invert LUT: ${job.invertLut ? "true" : "false"}` : null,
                  typeof job.copyMetadataRisky === "boolean"
                    ? `Copy metadata (risky): ${job.copyMetadataRisky ? "true" : "false"}`
                    : null,
                  datasetLines ? `Datasets:\n${datasetLines}` : null,
                ].filter(Boolean);

                return (
                  <div key={job.jobId} className={`sidebarJobTab is-${job.status}`}>
                    <div className="sidebarJobTop">
                      <div className="sidebarJobTitle">{title}</div>
                      <div className="sidebarJobActions">
                        {job.status === "running" ? (
                          <button
                            type="button"
                            className="sidebarJobStop"
                            onClick={() => void jobs.stopJob(job.jobId)}
                            aria-label="Stop job"
                            title="Stop"
                          >
                            Stop
                          </button>
                        ) : (
                          <button
                            type="button"
                            className="sidebarJobClose"
                            onClick={() => void jobs.removeJob(job.jobId)}
                            aria-label="Close job"
                            title="Close"
                          >
                            ×
                          </button>
                        )}
                      </div>
                    </div>
                    <div className="sidebarJobMeta">
                      {job.total > 0 ? (
                        <span>
                          {doneText} · {pctValue}% · {elapsedText}
                          {remainingText ? ` · ~${remainingText}` : ""}
                        </span>
                      ) : (
                        <span>
                          {job.current ?? job.status} · {elapsedText}
                        </span>
                      )}
                    </div>
                    <div className="sidebarJobBar" aria-hidden="true">
                      <div className="sidebarJobBarFill" style={{ width: `${Math.max(2, pctValue)}%` }} />
                    </div>
                    {failureSummary ? <div className="sidebarJobError">{failureSummary}</div> : null}
                    {canInspect ? (
                      <div className="sidebarJobFooter">
                        <button
                          type="button"
                          className="sidebarJobDetailButton"
                          onClick={() => void loadJobLog(job.jobId, { keepData: false })}
                          disabled={jobLog.openJobId === job.jobId && jobLog.loading}
                        >
                          {detailsButtonLabel}
                        </button>
                      </div>
                    ) : null}
                    <div className="sidebarJobTooltip" role="tooltip" aria-hidden="true">
                      <div className="sidebarJobTooltipInner">{tooltipLines.join("\n")}</div>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : null}
        </SidebarSection>
      </div>

      <Modal
        open={jobLog.openJobId !== null}
        title={selectedJobTitle}
        onClose={closeJobLog}
        panelClassName="jobLogModalPanel"
        bodyClassName="jobLogModalBody"
        footer={
          <div className="jobLogModalFooter">
            <button
              type="button"
              className="sidebarActionButton"
              onClick={() => {
                if (jobLog.openJobId) void loadJobLog(jobLog.openJobId);
              }}
              disabled={!jobLog.openJobId || jobLog.loading}
            >
              {jobLog.loading ? "Refreshing..." : "Refresh log"}
            </button>
          </div>
        }
      >
        {jobLog.error ? <div className="sidebarError">{jobLog.error}</div> : null}
        {!jobLog.data && jobLog.loading ? <div className="sidebarHint">Loading job details...</div> : null}
        {jobLog.data ? (
          <>
            <div className="jobLogMetaGrid">
              <div className="jobLogMetaCard">
                <div className="jobLogMetaLabel">Status</div>
                <div className="jobLogMetaValue">
                  {jobLog.data.status}
                  {jobLog.data.current ? ` · ${jobLog.data.current}` : ""}
                </div>
              </div>
              <div className="jobLogMetaCard">
                <div className="jobLogMetaLabel">Exit code</div>
                <div className="jobLogMetaValue">
                  {typeof jobLog.data.exitCode === "number" ? jobLog.data.exitCode : "n/a"}
                </div>
              </div>
              <div className="jobLogMetaCard">
                <div className="jobLogMetaLabel">Log lines</div>
                <div className="jobLogMetaValue">
                  {jobLog.data.truncated
                    ? `Showing last ${jobLog.data.logLines.length} of ${jobLog.data.totalLogLines}`
                    : `${jobLog.data.totalLogLines} captured`}
                </div>
              </div>
            </div>

            {formatFailureSummary(jobLog.data) ? (
              <div className="sidebarJobError isExpanded">{formatFailureSummary(jobLog.data)}</div>
            ) : null}

            <div className="jobLogSection">
              <div className="jobLogSectionTitle">Working directory</div>
              <div className="jobLogPath">{jobLog.data.workingDirectory ?? "Unavailable"}</div>
            </div>

            <div className="jobLogSection">
              <div className="jobLogSectionTitle">Log file</div>
              <div className="jobLogPath">{jobLog.data.logPath ?? "In-memory log only"}</div>
            </div>

            <div className="jobLogSection">
              <div className="jobLogSectionTitle">Command</div>
              {jobLog.data.command ? (
                <pre className="jobLogPre">{jobLog.data.command}</pre>
              ) : (
                <div className="sidebarHint">Command metadata was not captured for this job.</div>
              )}
            </div>

            <div className="jobLogSection">
              <div className="jobLogSectionTitle">Log tail</div>
              {jobLog.data.logLines.length > 0 ? (
                <pre className="jobLogPre">{jobLog.data.logLines.join("\n")}</pre>
              ) : (
                <div className="sidebarHint">No process output was captured for this job.</div>
              )}
            </div>
          </>
        ) : null}
      </Modal>
    </aside>
  );
}

function formatDurationMs(totalMs: number): string {
  const ms = Math.max(0, Math.floor(totalMs));
  const totalSeconds = Math.floor(ms / 1000);
  const seconds = totalSeconds % 60;
  const totalMinutes = Math.floor(totalSeconds / 60);
  const minutes = totalMinutes % 60;
  const hours = Math.floor(totalMinutes / 60);
  const pad2 = (value: number) => String(value).padStart(2, "0");
  if (hours > 0) return `${hours}:${pad2(minutes)}:${pad2(seconds)}`;
  return `${minutes}:${pad2(seconds)}`;
}

function estimateRemaining(status: string, elapsedMs: number, processed: number, total: number): string | null {
  if (status !== "running") return null;
  if (!(total > 0)) return null;
  if (!(processed > 0)) return null;
  if (processed < 5) return null;
  if (elapsedMs < 1500) return null;
  const perItem = elapsedMs / processed;
  const remaining = Math.max(0, Math.round((total - processed) * perItem));
  return formatDurationMs(remaining);
}

function formatFailureSummary(job: Pick<JobSummary, "status" | "error" | "exitCode">): string | null {
  if (job.status !== "failed") return null;
  const errorText = job.error?.trim() ?? "";
  if (errorText) {
    if (typeof job.exitCode === "number" && job.exitCode !== 0) {
      return `Exit ${job.exitCode}. ${errorText}`;
    }
    return errorText;
  }
  if (typeof job.exitCode === "number") {
    return `Exit ${job.exitCode}.`;
  }
  return "Job failed.";
}

function SidebarSection({
  title,
  subtitle,
  right,
  children,
}: {
  title: string;
  subtitle?: string | null;
  right?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="sidebarSection" aria-label={title}>
      <div className="sidebarSectionHeader">
        <div className="sidebarSectionTitle">{title}</div>
        {right ? <div className="sidebarSectionRight">{right}</div> : null}
      </div>
      {subtitle ? <div className="sidebarSectionSubtitle">{subtitle}</div> : null}
      <div className="sidebarSectionBody">{children}</div>
    </section>
  );
}

function SidebarActionButton({
  onClick,
  disabled,
  children,
}: {
  onClick: () => void | Promise<void>;
  disabled: boolean;
  children: ReactNode;
}) {
  return (
    <button type="button" className="sidebarActionButton" onClick={onClick} disabled={disabled}>
      {children}
    </button>
  );
}

function formatTime(date: Date) {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function pct(used: number, total: number) {
  if (total <= 0) return 0;
  return Math.round((used / total) * 100);
}

function formatJobTitle(type: string) {
  const normalized = type.trim();
  if (!normalized) return "Job";
  return normalized
    .split(/[-_]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}
