"""
iAODE: Interpretable Autoencoder with Ordinary Differential Equations

A deep learning framework for single-cell omics data analysis combining
variational autoencoders with neural ODEs for trajectory inference and
dimensionality reduction.

Main Components:
- agent: High-level interface for model training and evaluation
- annotation: Peak-to-gene annotation pipeline for scATAC-seq data
- BEN: Benchmark evaluation framework for comparing dimensionality
       reduction methods
- environment: Data handling and train/validation/test splitting
- model: Core VAE model with ODE integration
- module: Neural network building blocks (encoders, decoders, ODE functions)
- utils: Utility functions for data preprocessing
- DRE: Dimensionality reduction quality evaluation metrics
- LSE: Latent space evaluation for single-cell data

Usage Example:
    >>> import anndata as ad
    >>> import iaode
    >>>
    >>> # Load your data
    >>> adata = ad.read_h5ad('your_data.h5ad')
    >>>
    >>> # Create and train model
    >>> model = iaode.agent(
    ...     adata,
    ...     layer='counts',
    ...     latent_dim=10,
    ...     use_ode=True
    ... )
    >>> model.fit(epochs=100)
    >>>
    >>> # Get latent representation
    >>> latent = model.get_latent()
"""

__version__ = "0.3.0"
__author__ = "Zeyu Fu"

# Import main classes and functions
from .agent import agent
from .annotation import (
    annotation_pipeline,
    load_10x_h5_data,
    add_peak_coordinates,
    annotate_peaks_to_genes,
)
from .BEN import (
    DataSplitter,
    train_scvi_models,
    evaluate_scvi_models,
)
from .DRE import (
    DimensionalityReductionEvaluator,
    evaluate_dimensionality_reduction,
    compare_dimensionality_reduction_methods,
)
from .LSE import (
    SingleCellLatentSpaceEvaluator,
    evaluate_single_cell_latent_space,
    compare_single_cell_methods,
)
from .utils import (
    tfidf_normalization,
    select_highly_variable_peaks,
)

# Import datasets module (lazy import to avoid unnecessary dependencies)
from . import datasets

# Define public API
__all__ = [
    # Main agent class
    'agent',
    
    # Annotation pipeline
    'annotation_pipeline',
    'load_10x_h5_data',
    'add_peak_coordinates',
    'annotate_peaks_to_genes',
    
    # Benchmark evaluation
    'DataSplitter',
    'train_scvi_models',
    'evaluate_scvi_models',
    
    # Evaluation metrics
    'DimensionalityReductionEvaluator',
    'evaluate_dimensionality_reduction',
    'compare_dimensionality_reduction_methods',
    'SingleCellLatentSpaceEvaluator',
    'evaluate_single_cell_latent_space',
    'compare_single_cell_methods',
    
    # Utility functions
    'tfidf_normalization',
    'select_highly_variable_peaks',
    
    # Datasets module
    'datasets',
]
