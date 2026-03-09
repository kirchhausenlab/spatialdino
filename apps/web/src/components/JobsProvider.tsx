import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";

import type { JobSummary } from "../api/jobs";
import { cancelJob, clearFinishedJobs, listJobs, removeJob } from "../api/jobs";

type JobsContextValue = {
  jobs: JobSummary[];
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  stopJob: (jobId: string) => Promise<void>;
  removeJob: (jobId: string) => Promise<void>;
  clearFinished: () => Promise<void>;
};

const JobsContext = createContext<JobsContextValue | null>(null);

export function useJobs(): JobsContextValue {
  const ctx = useContext(JobsContext);
  if (!ctx) throw new Error("useJobs must be used within <JobsProvider>.");
  return ctx;
}

export default function JobsProvider({ children }: { children: ReactNode }) {
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollTimerRef = useRef<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const refresh = useCallback(async () => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setLoading(true);
    setError(null);
    try {
      const data = await listJobs({ signal: controller.signal });
      setJobs(data.jobs);
    } catch (err) {
      if (controller.signal.aborted) return;
      const message = err instanceof Error ? err.message : "Unknown error";
      setError(message);
    } finally {
      if (!controller.signal.aborted) setLoading(false);
    }
  }, []);

  const schedule = useCallback(
    (delayMs: number) => {
      if (pollTimerRef.current) window.clearTimeout(pollTimerRef.current);
      pollTimerRef.current = window.setTimeout(() => {
        void refresh();
      }, delayMs);
    },
    [refresh]
  );

  useEffect(() => {
    void refresh();
    return () => {
      abortRef.current?.abort();
      if (pollTimerRef.current) window.clearTimeout(pollTimerRef.current);
    };
  }, [refresh]);

  useEffect(() => {
    const hasRunning = jobs.some((job) => job.status === "running");
    if (hasRunning) {
      schedule(1000);
      return;
    }
    if (pollTimerRef.current) {
      window.clearTimeout(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  }, [jobs, schedule]);

  const stopJob = useCallback(
    async (jobId: string) => {
      await cancelJob(jobId);
      void refresh();
    },
    [refresh]
  );

  const removeJobById = useCallback(
    async (jobId: string) => {
      await removeJob(jobId);
      void refresh();
    },
    [refresh]
  );

  const clearFinished = useCallback(async () => {
    await clearFinishedJobs();
    void refresh();
  }, [refresh]);

  const value = useMemo(
    () => ({ jobs, loading, error, refresh, stopJob, removeJob: removeJobById, clearFinished }),
    [clearFinished, error, jobs, loading, refresh, removeJobById, stopJob]
  );

  return <JobsContext.Provider value={value}>{children}</JobsContext.Provider>;
}
