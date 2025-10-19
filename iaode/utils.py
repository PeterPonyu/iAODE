
"""
Evaluation metrics and utility functions for clustering and dimensionality reduction analysis.

This module provides comprehensive evaluation metrics for clustering performance,
graph connectivity analysis, and various statistical measures.
"""

import numpy as np
from numpy import ndarray
import pandas as pd
import scib
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import (
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from scipy.sparse import csr_matrix
from scipy.sparse import csgraph
from scipy.sparse import issparse


def get_dfs(mode, agent_list):
    """
    Generate DataFrames from agent evaluation results.
    
    Processes a list of agents and their scores to compute either mean or standard
    deviation statistics across multiple runs, returning DataFrames with standardized
    metric columns.
    
    Parameters
    ----------
    mode : str
        Statistical operation mode. Either "mean" for average scores or "std" 
        for standard deviation across runs.
    agent_list : list
        List of agent objects containing score attributes with evaluation metrics.
        
    Returns
    -------
    generator
        Generator yielding pandas DataFrames with columns: 
        ["ARI", "NMI", "ASW", "C_H", "D_B", "P_C"]
        
    Raises
    ------
    ValueError
        If mode is not "mean" or "std".
    """
    if mode == "mean":
        # Compute mean across all agent runs
        ls = list(
            map(
                lambda x: zip(
                    *(
                        np.array(b).mean(axis=0)
                        for b in zip(*((zip(*a.score)) for a in x))
                    )
                ),
                list(zip(*agent_list)),
            )
        )
    elif mode == "std":
        # Compute standard deviation across all agent runs
        ls = list(
            map(
                lambda x: zip(
                    *(
                        np.array(b).std(axis=0)
                        for b in zip(*((zip(*a.score)) for a in x))
                    )
                ),
                list(zip(*agent_list)),
            )
        )
    else:
        raise ValueError("Mode must be either 'mean' or 'std'")
    
    # Convert results to DataFrames with standardized column names
    return map(
        lambda x: pd.DataFrame(x, columns=["ARI", "NMI", "ASW", "C_H", "D_B", "P_C"]),
        ls,
    )


def moving_average(a, window_size):
    """
    Calculate moving average with specified window size.
    
    Applies a centered rolling window to smooth the input array, handling
    edge cases by using minimum periods of 1.
    
    Parameters
    ----------
    a : array-like
        Input array for which to calculate moving average.
    window_size : int
        Size of the rolling window for averaging.
        
    Returns
    -------
    numpy.ndarray
        Array of same length as input with smoothed values.
    """
    series = pd.Series(a)
    return (
        series.rolling(window=window_size, center=True, min_periods=1).mean().to_numpy()
    )


def fetch_score(adata1, q_z, label_true, label_mode="KMeans", batch=False):
    """
    Compute comprehensive clustering evaluation metrics.
    
    Evaluates clustering quality using multiple metrics including mutual information,
    silhouette analysis, and graph connectivity. Supports different clustering
    strategies and optional batch effect analysis.
    
    Parameters
    ----------
    adata1 : anndata.AnnData
        Annotated data matrix containing observations and metadata.
    q_z : numpy.ndarray
        Latent representation matrix of shape (n_samples, n_features).
    label_true : array-like
        Ground truth cluster labels for evaluation.
    label_mode : str, default="KMeans"
        Clustering strategy. Options: "KMeans", "Max", "Min".
        - "KMeans": Apply K-means clustering
        - "Max": Assign labels based on maximum values
        - "Min": Assign labels based on minimum values
    batch : bool, default=False
        Whether to compute batch effect metrics (iLISI, batch ASW).
        
    Returns
    -------
    tuple
        If batch=False: (NMI, ARI, ASW, C_H, D_B)
        If batch=True: (NMI, ARI, ASW, C_H, D_B, G_C, cLISI, iLISI, bASW)
        
        - NMI: Normalized Mutual Information
        - ARI: Adjusted Rand Index  
        - ASW: Average Silhouette Width
        - C_H: Calinski-Harabasz Index
        - D_B: Davies-Bouldin Index
        - G_C: Graph Connectivity
        - cLISI: Cell-type Local Inverse Simpson's Index
        - iLISI: Integration Local Inverse Simpson's Index
        - bASW: Batch-corrected Average Silhouette Width
        
    Raises
    ------
    ValueError
        If label_mode is not one of "KMeans", "Max", "Min".
    """
    # Subsample if dataset is too large for efficient computation
    if adata1.shape[0] > 3e3:
        idxs = np.random.choice(
            np.random.permutation(adata1.shape[0]), 3000, replace=False
        )
        adata1 = adata1[idxs, :]
        q_z = q_z[idxs, :]
        label_true = label_true[idxs]
    
    # Generate cluster labels based on specified mode
    if label_mode == "KMeans":
        labels = KMeans(q_z.shape[1]).fit_predict(q_z)
    elif label_mode == "Max":
        labels = np.argmax(q_z, axis=1)
    elif label_mode == "Min":
        labels = np.argmin(q_z, axis=1)
    else:
        raise ValueError("Mode must be one of 'KMeans', 'Max', or 'Min'")

    # Store results in AnnData object
    adata1.obsm["X_qz"] = q_z
    adata1.obs["label"] = pd.Categorical(labels)

    # Calculate core clustering metrics
    NMI = normalized_mutual_info_score(label_true, labels)
    ARI = adjusted_mutual_info_score(label_true, labels)
    ASW = silhouette_score(q_z, labels)
    
    # Take absolute value for non-KMeans methods to ensure positive ASW
    if label_mode != "KMeans":
        ASW = abs(ASW)
        
    C_H = calinski_harabasz_score(q_z, labels)
    D_B = davies_bouldin_score(q_z, labels)

    # Further subsample for graph-based metrics if needed
    if adata1.shape[0] > 5e3:
        idxs = np.random.choice(
            np.random.permutation(adata1.shape[0]), 5000, replace=False
        )
        adata1 = adata1[idxs, :]
    
    # Calculate graph connectivity and LISI metrics
    G_C = graph_connection(
        kneighbors_graph(adata1.obsm["X_qz"], 15), adata1.obs["label"].values
    )
    clisi = scib.metrics.clisi_graph(adata1, "label", "embed", "X_qz", n_cores=-2)
    
    # Include batch effect metrics if requested
    if batch:
        ilisi = scib.metrics.ilisi_graph(adata1, "batch", "embed", "X_qz", n_cores=-2)
        bASW = scib.metrics.silhouette_batch(adata1, "batch", "label", "X_qz")
        return NMI, ARI, ASW, C_H, D_B, G_C, clisi, ilisi, bASW
    
    return NMI, ARI, ASW, C_H, D_B


def graph_connection(graph: csr_matrix, labels: ndarray):
    """
    Calculate graph connectivity metric for clustering evaluation.
    
    Measures how well-connected clusters are by analyzing the largest connected
    component within each cluster subgraph. Higher values indicate better
    cluster connectivity.
    
    Parameters
    ----------
    graph : scipy.sparse.csr_matrix
        Sparse adjacency matrix representing the neighborhood graph.
    labels : numpy.ndarray
        Cluster assignment labels for each node in the graph.
        
    Returns
    -------
    float
        Average connectivity score across all clusters, ranging from 0 to 1.
        Values closer to 1 indicate better cluster connectivity.
    """
    cg_res = []
    
    # Analyze connectivity for each unique cluster
    for l in np.unique(labels):
        # Extract subgraph for current cluster
        mask = np.where(labels == l)[0]
        subgraph = graph[mask, :][:, mask]
        
        # Find connected components within cluster
        _, lab = csgraph.connected_components(subgraph, connection="strong")
        
        # Calculate connectivity as ratio of largest component to total nodes
        tab = np.unique(lab, return_counts=True)[1]
        cg_res.append(tab.max() / tab.sum())
    
    return np.mean(cg_res)


def quiver_autoscale(E: np.ndarray, V: np.ndarray):
    """
    Automatically determine optimal scaling for quiver plots.
    
    Calculates appropriate scale factor for matplotlib quiver plots by
    creating a temporary plot and extracting the auto-computed scale,
    then normalizing by the position scale factor.
    
    Parameters
    ----------
    E : numpy.ndarray
        Position coordinates for quiver plot of shape (n_points, 2).
    V : numpy.ndarray  
        Vector field values of shape (n_points, 2).
        
    Returns
    -------
    float
        Optimal scale factor for quiver plot visualization.
    """
    import matplotlib.pyplot as plt

    # Create temporary figure for scale computation
    fig, ax = plt.subplots()
    scale_factor = np.abs(E).max()

    # Generate quiver plot with normalized positions
    Q = ax.quiver(
        E[:, 0] / scale_factor,
        E[:, 1] / scale_factor,
        V[:, 0],
        V[:, 1],
        angles="xy",
        scale=None,
        scale_units="xy",
    )
    
    # Initialize quiver to compute auto-scale
    Q._init()
    
    # Clean up temporary figure
    fig.clf()
    plt.close(fig)
    
    return Q.scale / scale_factor


def l2_norm(x, axis=-1):
    """
    Compute L2 (Euclidean) norm along specified axis.
    
    Efficiently calculates L2 norm for both dense and sparse matrices,
    automatically handling different input types.
    
    Parameters
    ----------
    x : array-like or scipy.sparse matrix
        Input array or sparse matrix for norm computation.
    axis : int, default=-1
        Axis along which to compute the norm. Default is last axis.
        
    Returns
    -------
    numpy.ndarray
        L2 norm values along the specified axis.
    """
    if issparse(x):
        # Efficient computation for sparse matrices
        return np.sqrt(x.multiply(x).sum(axis=axis).A1)
    else:
        # Standard computation for dense arrays
        return np.sqrt(np.sum(x * x, axis=axis))

