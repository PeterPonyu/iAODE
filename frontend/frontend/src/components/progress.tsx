"use client";

import { useEffect, useMemo, useState, useRef } from "react";
import { getTrainingState } from "@/lib/api";
import { TrainingState } from "@/lib/types";

type Props = {
    pollIntervals?: number;
    maxEpochs?: number;
    className?: string;
};

export default function TrainingProgress({
    pollIntervals = 2000,
    maxEpochs,
    className,
}: Props) {
    const [data, setData] = useState<TrainingState | null>(null);
    const [error, setError] = useState<string | null>(null);
    const timerRef = useRef<number | null>(null);

    const fetchState = async () => {
        try {
            const res = await getTrainingState();
            setData(res);
            setError(null);
        } catch (e: any) {
            setError(e?.message || "Failed to fetch training state");
        }
    };

    useEffect(() => {
        fetchState();
        timerRef.current = window.setInterval(fetchState, pollIntervals);
        return () => {
            if (timerRef.current) {
                window.clearInterval(timerRef.current);
                timerRef.current = null;
        }
    };
    }, [pollIntervals]);

    const progressPct = useMemo(() => {
        if (!data || !maxEpochs || maxEpochs <= 0) return null;
        return Math.round(Math.max(0, Math.min(100, (data.epoch / maxEpochs) * 100)));
    }, [data, maxEpochs]);

    const fmt = (v: number | undefined, digits = 4) =>
        typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "—";
    return (
            <div className={className}>
      {error && (
        <div className="mb-3 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-red-800">
          <strong className="font-semibold">Error:</strong> {error}
        </div>
      )}

      {!data && !error && (
        <div className="text-slate-500">Loading training state…</div>
      )}

      {data && (
        <div className="rounded-xl border border-slate-200 bg-white p-4">
            <div className="grid gap-3 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
                <Metric label="Epoch" value={String(data.epoch)} />
                <Metric label="ARI" value={fmt(data.ARI)} />
                <Metric label="NMI" value={fmt(data.NMI)} />
                <Metric label="ASW" value={fmt(data.ASW)} />
                <Metric label="CAL" value={fmt(data.CAL)} />
                <Metric label="DAV" value={fmt(data.DAV)} />
                <Metric label="COR" value={fmt(data.COR)} />
            </div>

          {typeof progressPct === "number" && (
            <div className="mt-4">
                <div className="mb-1 flex items-center justify-between text-slate-700">
                    <span>Epoch Progress</span>
                    <span className="tabular-nums">
                    {progressPct}% ({data.epoch}/{maxEpochs})
                    </span>
                </div>
              <ProgressBar id="epoch-progress" percent={progressPct} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
      <div className="text-xs text-slate-500">{label}</div>
      <div className="text-lg font-semibold text-slate-900">{value}</div>
    </div>
  );
}

function ProgressBar({
  id,
  percent,
}: {
  id?: string;
  percent: number; // 0..100
}) {
  const p = Number.isFinite(percent) ? Math.max(0, Math.min(100, percent)) : 0;
  return (
    <div className="w-full">
      <progress
        id={id}
        className="w-full h-3 overflow-hidden rounded-full [appearance:none]"
        value={p}
        max={100}
        aria-label="Training progress"
        title={`${p}%`}
      />
      <div className="sr-only" aria-live="polite">
        {p}% complete
      </div>
    </div>
  );
}