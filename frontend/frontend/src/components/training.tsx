"use client";

import { useState } from "react";
import TrainingParamsForm from "./params";
import TrainingProgress from "./progress";
import DownloadButtons from "./downloads";
import { AgentParams, TrainingParams } from "@/lib/types";
import { startTraining } from "@/lib/api"; 

export default function Training() {
  const [agent, setAgent] = useState<AgentParams | null>(null);
  const [training, setTraining] = useState<TrainingParams | null>(null);
  const [status, setStatus] = useState<"idle" | "starting" | "success" | "error">("idle");
  const [message, setMessage] = useState<string>("");

  const handleParams = (v: { agent: AgentParams; training: TrainingParams }) => {
    setAgent(v.agent);
    setTraining(v.training);
  };


  const handleStart = async () => {
    if (!agent || !training) return;
    try {
      setStatus("starting");
      setMessage("");
      const res = await startTraining(agent, training);
      setStatus("success");
      setMessage(res.message || "Training started");
    } catch (e: any) {
      setStatus("error");
      setMessage(e?.message || "Failed to start training");
    }
  };

  const started = status === "success";

  return (
    <main className="grid gap-4 p-4">
      <TrainingParamsForm onChange={handleParams} onSubmit={handleParams} />
      <div className="flex items-center gap-3">
        <button
          onClick={handleStart}
          disabled={!agent || !training || status === "starting"}
          className="inline-flex items-center rounded-md bg-indigo-600 hover:bg-indigo-800 px-4 py-2 text-white disabled:opacity-50"
        >
          {status === "starting" ? "Starting..." : "Start Training"}
        </button>
        {message && <p className="text-sm text-slate-700 m-0">{message}</p>}
      </div>
      {started && (
        <section className="mt-2 space-y-4">
          <div>
            <h2 className="mb-2 text-lg font-medium">Training Progress</h2>
            <TrainingProgress pollIntervals={2000} maxEpochs={training?.epochs as number | undefined} />
          </div>
          
          <div>
            <h2 className="mb-2 text-lg font-medium">Download Embeddings</h2>
            <DownloadButtons/>
          </div>
        </section>
      )}
    </main>
  );
}