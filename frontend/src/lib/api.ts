import { AgentParams, TrainingParams, TrainingState, DataInfo } from "./types";

const API_BASE_URL = 'http://localhost:8000'

export async function startTraining(
    agentParams: AgentParams,
    trainParams: TrainingParams
): Promise<{ message: string }> {
    const response = await fetch(`${API_BASE_URL}/train`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            agparams: agentParams,
            trainparams: trainParams
        }),
    });
    if (!response.ok) {
        throw new Error('Failed to start training');
    }
    return response.json();
}

export async function getTrainingState(): Promise<TrainingState> {
    const response = await fetch(`${API_BASE_URL}/state`);
    if (!response.ok) {
        throw new Error('Failed to fetch training state');
    }
    return response.json();
}

export async function uploadData(file: File): Promise<DataInfo> {
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData,
    })
    if (!response.ok) {
        throw new Error('Failed to upload data');
    }
    return response.json();
}

export async function getLatent(embeddingType: 'latent' | 'interpretable'): Promise<void> {
    const url = new URL(`${API_BASE_URL}/download`);

    if (embeddingType) {
        url.searchParams.set('embedding_type', embeddingType);
    }
    
    const response = await fetch(url.toString())

    if (!response.ok) {
        throw new Error('Failed to download latent embedding');
    }

    const blob = await response.blob();
    const tempUrl = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = tempUrl;

    // Extract filename from header if provided
    const cd = response.headers.get('content-disposition');
    let filename = `${embeddingType}_embedding.csv`;
    if (cd) {
        const m = cd.match(/filename="(.+)"/);
        if (m?.[1]) filename = m[1];
    }

    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(tempUrl);
}
