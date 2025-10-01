"user client"

import { useState } from "react";
import { getLatent } from "@/lib/api";

type EmbeddingType = 'latent' | 'interpretable';

export default function DownloadButtons({
    disabled,
    className,
}: {
    disabled?: boolean;
    className?: string;
}) {
    const [status, setStatus] = useState<"idle" | "downloading" | "error" | "success">("idle");
    const [message, setMessage] = useState<string>("");

    const handleDownload = async (type: EmbeddingType) => {
        if (disabled || status === "downloading") return;
        try {
            setStatus("downloading");
            setMessage("");
            await getLatent(type);
            setStatus("success");
            setMessage(`Download ${type} embedding`);
            setTimeout(() => {
                setStatus("idle");
                setMessage("");
            }, 2000);
        } catch (e: any) {
            setStatus("error");
            setMessage(e?.message || "Download failed");
        }
    };

    return (
        <div className={className}>
        <div className="flex flex-wrap items-center gap-2">
            <button
            onClick={() => handleDownload("latent")}
            disabled={disabled || status === "downloading"}
            className="inline-flex items-center rounded-md bg-slate-600 px-3 py-2 text-white disabled:opacity-50 hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-slate-900 focus:ring-offset-2"
            >
            {status === "downloading" ? "Downloading…" : "Download Latent"}
            </button>
            <button
            onClick={() => handleDownload("interpretable")}
            disabled={disabled || status === "downloading"}
            className="inline-flex items-center rounded-md bg-slate-600 px-3 py-2 text-white disabled:opacity-50 hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-slate-900 focus:ring-offset-2"
            >
            {status === "downloading" ? "Downloading…" : "Download Interpretable"}
            </button>
        </div>

        {message && (
            <p
            className={`mt-2 text-sm ${
                status === "error" ? "text-red-600" : "text-slate-700"
            }`}
            >
            {message}
            </p>
        )}
        </div>
    );
}