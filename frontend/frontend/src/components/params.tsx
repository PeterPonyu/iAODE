"use client";

import { useState } from "react";
import { AgentParams, TrainingParams } from "@/lib/types";

type Props = {
  initialAgent?: AgentParams;
  initialTraining?: TrainingParams;
  onChange?: (v: { agent: AgentParams; training: TrainingParams }) => void;
  onSubmit?: (v: { agent: AgentParams; training: TrainingParams }) => void;
};

const defaultAgent: AgentParams = {
  later: "counts",
  batch_percent: 0.02,
  recon: 1.0,
  irecon: 0.0,
  beta: 1.0,
  dip: 0.0,
  tc: 0.0,
  info: 0.0,
  hidden_dims: 128,
  latent_dim: 10,
  i_dim: 2,
  use_ode: false,
  loss_mode: "nb",
  lr: 1e-3,
  vae_reg: 0.5,
  ode_reg: 0.5,
};

const defaultTraining: TrainingParams = {
  epochs: 3000,
};

export default function TrainingParamsForm({
  initialAgent,
  initialTraining,
  onChange,
  onSubmit,
}: Props) {
  const [agent, setAgent] = useState<AgentParams>(initialAgent ?? defaultAgent);
  const [training, setTraining] = useState<TrainingParams>(
    initialTraining ?? defaultTraining
  );

  const update = (nextAgent: AgentParams, nextTraining: TrainingParams) => {
    setAgent(nextAgent);
    setTraining(nextTraining);
    onChange?.({ agent: nextAgent, training: nextTraining });
  };

  const handleNumber =
    <K extends keyof AgentParams>(key: K) =>
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const v = e.target.value;
      const n = v === "" ? ("" as unknown as number) : Number(v);
      update(
        { ...agent, [key]: Number.isNaN(n) ? 0 : n } as AgentParams,
        training
      );
    };

  const handleBool =
    <K extends keyof AgentParams>(key: K) =>
    (e: React.ChangeEvent<HTMLInputElement>) => {
      update({ ...agent, [key]: e.target.checked } as AgentParams, training);
    };

  const handleText =
    <K extends keyof AgentParams>(key: K) =>
    (e: React.ChangeEvent<HTMLInputElement>) => {
      update({ ...agent, [key]: e.target.value } as AgentParams, training);
    };

  const handleLoss = (e: React.ChangeEvent<HTMLSelectElement>) => {
    update(
      { ...agent, loss_mode: e.target.value as AgentParams["loss_mode"] },
      training
    );
  };

  const handleEpochs = (e: React.ChangeEvent<HTMLInputElement>) => {
    const n = Number(e.target.value);
    update(agent, { epochs: Number.isNaN(n) ? 0 : n });
  };

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit?.({ agent, training });
  };

  return (
    <form
      onSubmit={submit}
      className="mx-auto max-w-3xl space-y-8 rounded-lg border border-gray-200 bg-white p-6 shadow-sm"
    >
      {/* AgentParams */}
      <section>
        <h3 className="mb-4 text-lg font-semibold text-gray-900">Agent Parameters</h3>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          {/* later */}
          <FieldText
            id="later"
            label="Later"
            value={agent.later}
            onChange={handleText("later")}
            placeholder="Optional note"
          />

          <FieldNumber
            id="batch_percent"
            label="Batch Percent"
            value={agent.batch_percent}
            onChange={handleNumber("batch_percent")}
            step={0.001}
            min={0}
            max={1}
          />

          <FieldNumber
            id="recon"
            label="Recon"
            value={agent.recon}
            onChange={handleNumber("recon")}
            step={0.001}
            min={0}
          />

          <FieldNumber
            id="irecon"
            label="I-Recon"
            value={agent.irecon}
            onChange={handleNumber("irecon")}
            step={0.001}
            min={0}
          />

          <FieldNumber
            id="beta"
            label="Beta"
            value={agent.beta}
            onChange={handleNumber("beta")}
            step={0.001}
            min={0}
          />

          <FieldNumber
            id="dip"
            label="DIP"
            value={agent.dip}
            onChange={handleNumber("dip")}
            step={0.001}
            min={0}
          />

          <FieldNumber
            id="tc"
            label="TC"
            value={agent.tc}
            onChange={handleNumber("tc")}
            step={0.001}
            min={0}
          />

          <FieldNumber
            id="info"
            label="Info"
            value={agent.info}
            onChange={handleNumber("info")}
            step={0.001}
            min={0}
          />

          <FieldNumber
            id="hidden_dims"
            label="Hidden Dims"
            value={agent.hidden_dims}
            onChange={handleNumber("hidden_dims")}
            step={1}
            min={1}
          />

          <FieldNumber
            id="latent_dim"
            label="Latent Dim"
            value={agent.latent_dim}
            onChange={handleNumber("latent_dim")}
            step={1}
            min={1}
          />

          <FieldNumber
            id="i_dim"
            label="I Dim"
            value={agent.i_dim}
            onChange={handleNumber("i_dim")}
            step={1}
            min={0}
          />

          <FieldCheckbox
            id="use_ode"
            label="Use ODE"
            checked={agent.use_ode}
            onChange={handleBool("use_ode")}
            helper="Enable ODE-based dynamics"
          />

          <FieldSelect
            id="loss_mode"
            label="Loss Mode"
            value={agent.loss_mode}
            onChange={handleLoss}
            options={[
              { value: "mse", label: "MSE" },
              { value: "nb", label: "NB" },
              { value: "zinb", label: "ZINB" },
            ]}
          />

          <FieldNumber
            id="lr"
            label="Learning Rate"
            value={agent.lr}
            onChange={handleNumber("lr")}
            step={0.0001}
            min={0}
          />

          <FieldNumber
            id="vae_reg"
            label="VAE Reg"
            value={agent.vae_reg}
            onChange={handleNumber("vae_reg")}
            step={0.001}
            min={0}
          />

          <FieldNumber
            id="ode_reg"
            label="ODE Reg"
            value={agent.ode_reg}
            onChange={handleNumber("ode_reg")}
            step={0.001}
            min={0}
          />
        </div>
      </section>

      {/* TrainingParams */}
      <section>
        <h3 className="mb-4 text-lg font-medium text-gray-800">Training Settings</h3>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <FieldNumber
            id="epochs"
            label="Epochs"
            value={training.epochs}
            onChange={handleEpochs}
            step={1}
            min={1}
          />
        </div>
      </section>

      <div className="flex items-center gap-3">
        <button
          type="submit"
          className="inline-flex items-center rounded-md bg-gray-900 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2"
        >
          Apply
        </button>

        <button
          type="button"
          className="inline-flex items-center rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-gray-300 focus:ring-offset-2"
          onClick={() => {
            update(defaultAgent, defaultTraining);
          }}
        >
          Reset
        </button>
      </div>
    </form>
  );
}

/* --- Small field components for consistency --- */

type FieldBaseProps = {
  id: string;
  label: string;
  helper?: string;
};

function FieldWrapper({
  id,
  label,
  helper,
  children,
}: React.PropsWithChildren<FieldBaseProps>) {
  return (
    <div className="flex flex-col">
      <label htmlFor={id} className="mb-1 text-sm font-medium text-gray-700">
        {label}
      </label>
      {children}
      {helper ? <p className="mt-1 text-xs text-gray-500">{helper}</p> : null}
    </div>
  );
}

function baseInputClasses(error = false) {
  return [
    "block w-full rounded-md border px-3 py-2 text-sm shadow-sm focus:outline-none",
    error
      ? "border-red-300 focus:border-red-400 focus:ring-2 focus:ring-red-200"
      : "border-gray-300 focus:border-gray-400 focus:ring-2 focus:ring-gray-100",
  ].join(" ");
}

type FieldTextProps = FieldBaseProps & {
  value: string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  placeholder?: string;
};

function FieldText({ id, label, helper, value, onChange, placeholder }: FieldTextProps) {
  return (
    <FieldWrapper id={id} label={label} helper={helper}>
      <input
        id={id}
        type="text"
        className={baseInputClasses()}
        value={value ?? ""}
        onChange={onChange}
        placeholder={placeholder}
        autoComplete="off"
      />
    </FieldWrapper>
  );
}

type FieldNumberProps = FieldBaseProps & {
  value: number;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  step?: number;
  min?: number;
  max?: number;
};

function FieldNumber({
  id,
  label,
  helper,
  value,
  onChange,
  step,
  min,
  max,
}: FieldNumberProps) {
  return (
    <FieldWrapper id={id} label={label} helper={helper}>
      <input
        id={id}
        type="number"
        className={baseInputClasses()}
        value={Number.isFinite(value) ? value : 0}
        onChange={onChange}
        step={step}
        min={min}
        max={max}
        inputMode="decimal"
      />
    </FieldWrapper>
  );
}

type FieldCheckboxProps = FieldBaseProps & {
  checked: boolean;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
};

function FieldCheckbox({ id, label, helper, checked, onChange }: FieldCheckboxProps) {
  return (
    <div className="flex items-center gap-3">
      <input
        id={id}
        type="checkbox"
        className="h-4 w-4 rounded border-gray-300 text-gray-900 focus:ring-gray-300"
        checked={checked}
        onChange={onChange}
      />
      <label htmlFor={id} className="text-sm font-medium text-gray-700">
        {label}
      </label>
      {helper ? <p className="ml-2 text-xs text-gray-500">{helper}</p> : null}
    </div>
  );
}

type FieldSelectProps = FieldBaseProps & {
  value: string;
  onChange: (e: React.ChangeEvent<HTMLSelectElement>) => void;
  options: { value: string; label: string }[];
};

function FieldSelect({ id, label, helper, value, onChange, options }: FieldSelectProps) {
  return (
    <FieldWrapper id={id} label={label} helper={helper}>
      <select
        id={id}
        className={baseInputClasses()}
        value={value}
        onChange={onChange}
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </FieldWrapper>
  );
}