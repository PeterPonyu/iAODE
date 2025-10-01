"use client";

import React, { useMemo, useState, useRef } from "react";
import { uploadData } from "@/lib/api";

export default function FileUpload() {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<"idle" | "uploading" | "success" | "error">("idle");
  const [message, setMessage] = useState("");
  const [ncells, setCells] = useState<number>();
  const [ngenes, setGenes] = useState<number>();
  const inputRef = useRef<HTMLInputElement | null>(null);

  const onSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] || null;
    if (f && !f.name.toLowerCase().endsWith(".h5ad")) {
      setFile(null);
      setMessage("Please select a .h5ad file.");
      setStatus("error");
      return;
    }
    setFile(f);
    setMessage("");
    setStatus("idle");
  };

  const onUpload = async () => {
    if (!file) return;
    try {
      setStatus("uploading");
      setMessage("");
      const res = await uploadData(file);
      setCells(res.n_cells);
      setGenes(res.n_genes);
      setStatus("success");
      setMessage("Upload successful");
    } catch (err: any) {
      setStatus("error");
      setMessage(err?.message || "Upload failed");
    }
  };

  const reset = () => {
    setFile(null);
    setMessage("");
    setStatus("idle");
    setCells(undefined);
    setGenes(undefined);
    console.log(inputRef.current);
    if (inputRef.current) {
      inputRef.current.value = "";
    }
  };

  const fileSize = useMemo(() => {
    if (!file) return "";
    const kb = file.size / 1024;
    return kb < 1024 ? `${kb.toFixed(1)} KB` : `${(kb / 1024).toFixed(2)} MB`;
  }, [file]);

  const isUploading = status === "uploading";
  const canUpload = !!file && !isUploading;

  return (
    <div className="mx-auto w-full max-w-xl rounded-lg border border-gray-200 bg-white p-5 shadow-sm">
      <h2 className="text-lg font-semibold text-gray-900">Upload Dataset</h2>
      <p className="mt-1 mb-4 text-sm text-gray-500">Accepted format: .h5ad</p>

      {/* File input */}
      <label
        htmlFor="file"
        className="flex cursor-pointer items-center justify-between rounded-md border border-dashed border-gray-300 bg-gray-50 px-4 py-3 hover:border-gray-400"
      >
        <div className="min-w-0">
          <p className="text-sm font-medium text-gray-800">
            {file ? file.name : "Choose a .h5ad file"}
          </p>
          <p className="text-xs text-gray-500">
            {file ? `Size: ${fileSize}` : "Click to browse"}
          </p>
        </div>
        <span className="rounded-md bg-gray-900 px-3 py-1.5 text-xs font-medium text-white">
          Browse
        </span>
        <input
          id="file"
          type="file"
          accept=".h5ad"
          onChange={onSelect}
          disabled={isUploading}
          className="sr-only"
        />
      </label>

      {/* Actions */}
      <div className="mt-4 flex gap-3">
        <button
          type="button"
          onClick={onUpload}
          disabled={!canUpload}
          className={`inline-flex items-center rounded-md px-4 py-2 text-sm font-medium shadow-sm focus:outline-none focus:ring-2 focus:ring-gray-300 focus:ring-offset-2 ${
            canUpload ? "bg-gray-900 text-white hover:bg-gray-800" : "cursor-not-allowed bg-gray-200 text-gray-500"
          }`}
        >
          {isUploading ? "Uploading..." : "Upload"}
        </button>

        <button
          type="button"
          onClick={reset}
          disabled={isUploading || (!file && !message && !ncells && !ngenes)}
          className={`inline-flex items-center rounded-md border px-4 py-2 text-sm font-medium focus:outline-none focus:ring-2 focus:ring-gray-200 focus:ring-offset-2 ${
            isUploading || (!file && !message && !ncells && !ngenes)
              ? "cursor-not-allowed border-gray-200 bg-gray-100 text-gray-400"
              : "border-gray-300 bg-white text-gray-700 hover:bg-gray-50"
          }`}
        >
          Reset
        </button>
      </div>

      {/* Message */}
      {message && (
        <div
          className={`mt-4 rounded-md px-3 py-2 text-sm ring-1 ${
            status === "error"
              ? "bg-red-50 text-red-700 ring-red-200"
              : status === "success"
              ? "bg-green-50 text-green-700 ring-green-200"
              : "bg-blue-50 text-blue-700 ring-blue-200"
          }`}
        >
          {message}
        </div>
      )}

      {/* Result */}
      {typeof ncells === "number" && typeof ngenes === "number" && (
        <div className="mt-3 rounded-md border border-gray-200 bg-white px-3 py-2 text-sm text-gray-700">
          <span className="font-medium text-gray-900">Shape:</span> {ncells} cells × {ngenes} genes
        </div>
      )}
    </div>
  );
}