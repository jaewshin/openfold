# Retrieval-Augmented OpenFold SoloSeq: Project Report

## 1. Executive Summary
This project extends **OpenFold SoloSeq** (single-sequence protein structure prediction) with a **retrieval-augmented strategy**.  
Instead of relying only on the query sequence embedding, the model retrieves related proteins from large external databases and uses them to update the query representation before structure prediction.

The goal is to improve structure prediction quality while keeping the system practical at scale.

## 2. Core Problem and Motivation
OpenFold SoloSeq is efficient but does not directly use multi-sequence alignment context.  
This project adds retrieval as an external memory mechanism:

1. Encode the query sequence for retrieval.
2. Retrieve top-k neighbors from one or more FAISS indices.
3. Fuse retrieved information into query embeddings.
4. Feed the updated representation into OpenFold.
5. Train retrieval and selected OpenFold modules end-to-end.

This gives a controllable middle ground between pure single-sequence prediction and full MSA-heavy pipelines.

## 3. System Overview
The full system has six major components:

1. **Dataset preparation**
2. **Embedding generation**
3. **FAISS index construction**
4. **Retriever + fusion model**
5. **Training/finetuning pipeline**
6. **Testing and experiment orchestration**

Each component is designed to support large-scale datasets and multiple experiment variants.

## 4. Data Component
Two dataset paths are used in this project.

### 4.1 CATH/PDB-derived training fixture
Data is prepared from structure resources into train/validation splits with associated sequences and structures.

Outputs are in the standard training layout:
- `train.fasta`
- `val.fasta`
- `splits.json`
- structure files
- manifest metadata

### 4.2 OpenProteinNet path
A second workflow downloads and processes OpenProteinNet-style data:

1. Download structure assets from public storage.
2. Convert into retrieval-training format.
3. Build train/val split (including leakage-aware options via sequence clustering).
4. Generate query embeddings (ESM1b) for training inputs.

This enables larger-scale experiments and better split hygiene.

## 5. Embeddings and Indexing
Two external retrieval spaces are used:

1. **Sequence index**: built from ESM2-35M embeddings (dimension 480)
2. **Structure index**: built from TMVec-2s embeddings (dimension 512)

Both are stored as FAISS indices with row-aligned ID maps for lookup and evaluation.

Important design detail:
- FAISS indices are treated as fixed retrieval databases during model training.
- Retriever query encoders are trainable, but the index contents are not updated online (for cost reasons).

## 6. Retrieval Pipelines Implemented
The project supports three retrieval modes.

### 6.1 Legacy pipeline
- Query is derived from pooled ESM1b input embedding.
- Linear projections map query to FAISS spaces.
- Retrieved vectors are projected back to model embedding dimension and fused.

This is the baseline for backward compatibility.

### 6.2 Pipeline A: `embed_project`
- Uses explicit trainable query encoders matching index spaces:
  - ESM2 retriever encoder for sequence index
  - TMVec-2s retriever encoder for structure index
- Retrieves top-k vectors from FAISS.
- Projects retrieved vectors to OpenFold input dimension.
- Applies cross-attention fusion to update query tokens.

This pipeline directly aligns retriever encoders with how indices were built.

### 6.3 Pipeline B: `rawseq_esm1b`
- Uses the same explicit trainable query encoders as Pipeline A.
- After retrieval, converts FAISS row IDs to raw sequences.
- Embeds retrieved raw sequences with frozen ESM1b.
- Cross-attends over retrieved per-residue embeddings to update the query.

This is more biologically interpretable (sequence-level retrieval), but more memory/time intensive.

## 7. Fusion and Injection Strategy
Retrieved information is fused with query tokens via cross-attention and source-weight mixing.

When both sequence and structure retrievers are active:
- each source produces a fused representation
- a learned source-mix weighting combines them

Retrieval-conditioned updates can be injected at configurable stages:

1. **Input injection**: replace/update query embedding before OpenFold input embedder
2. **Pre-Evoformer injection**: additive update before Evoformer stack
3. **Pre-Structure injection**: additive update before Structure Module

This enables ablations on where retrieval helps most.

## 8. Finetuning Approaches
The project explores selective unfreezing of OpenFold modules while keeping retrievers trainable.

Common experiment variants include:

1. Frozen OpenFold + trainable retriever/fusion
2. Finetune Evoformer + retriever
3. Finetune Structure Module + retriever
4. Finetune Evoformer + Structure Module + retriever
5. Finetune Input Embedder + retriever
6. Finetune Input Embedder + Evoformer + retriever
7. Finetune full OpenFold + retriever

This experiment matrix is repeated for multiple pipelines and datasets.

## 9. Training Pipeline
Training is implemented with PyTorch Lightning and includes:

- configurable learning rate and optimizer settings
- per-device batch size + gradient accumulation (effective batch size control)
- validation interval control (e.g., every 1000 steps)
- optional Weights & Biases logging
- Slurm launch scripts for multi-run experiment management

The setup is designed for GPU cluster usage and large-scale run orchestration.

## 10. Memory and Scale Considerations
The project explicitly handles memory pressure from:
- large FAISS indices
- retrieval encoders
- OpenFold model
- optional embedding of retrieved sequences

Mitigations include:
- lazy FAISS loading
- CPU placement options for retrieved-sequence embedding
- gradient accumulation for large effective batch size with small per-GPU batch
- packed dataset support to reduce runtime preprocessing overhead

## 11. Validation and Testing
Testing covers:

1. retrieval/fusion module behavior
2. data parser and packed dataset integrity
3. gradient propagation to retriever encoders
4. end-to-end lightning smoke tests for multiple pipelines
5. FAISS retrieval sanity checks (row-level and ID-level)

This ensures both algorithmic correctness and training stability.

## 12. Current Research Direction
This is now a configurable research platform rather than a single model variant.  
The main question being evaluated is:

**Which combination of retrieval design, injection stage, and finetuning scope yields the best quality-to-compute tradeoff for structure prediction?**

The project is set up to answer this through controlled ablations across:
- retrieval source usage (sequence only / structure only / both)
- retrieval pipeline (legacy / embed-project / rawseq-ESM1b)
- injection stage (input / pre-evoformer / pre-structure)
- finetuning scope (frozen to full OpenFold)
