# Retrieval Flux Submission Matrix

This folder contains Flux submission wrappers for 3 retrieval pipelines x 7 finetuning modes.

Pipeline mapping used in these scripts:
- `pipeline_a` -> `--retrieval_pipeline legacy`
- `pipeline_b` -> `--retrieval_pipeline embed_project`
- `pipeline_c` -> `--retrieval_pipeline rawseq_esm1b`

Run matrix (per pipeline):
1. frozen OpenFold + train retriever/fusion
2. finetune structure module + train retriever/fusion
3. finetune evoformer + train retriever/fusion
4. finetune input embedder + train retriever/fusion
5. finetune input embedder + evoformer + train retriever/fusion
6. finetune evoformer + structure module + train retriever/fusion
7. finetune all OpenFold + train retriever/fusion

Each wrapper delegates to:
- `experiments/retrieval_flux/submit_retrieval_matrix_job.sh`
- `scripts/retrieval/flux_train_retrieval_pipeline_submit.sh`

Example:
```bash
bash experiments/retrieval_flux/run_pipeline_b_03_finetune_evoformer_trainable_retriever.sh
```

Useful env overrides:
- `FLUX_QUEUE`, `FLUX_TIME_LIMIT`, `FLUX_CORES`, `FLUX_GPUS`
- `DATA_ROOT`, `READY_DIR`, `SEQ_EMB_DIR`
- `USE_WANDB=1`, `WANDB_ENTITY=...`

## Multi-day continuation (dependent chain)

Use:

```bash
bash experiments/retrieval_flux/queue_retrieval_resume_pipeline.sh \
  --pipeline b \
  --run-id 05 \
  --num-jobs 5 \
  --queue pbatch \
  --time-limit 24h \
  --max-time 00:23:50:00
```

This enables:
- shared output directory per `run-tag`
- checkpoint auto-resume (`last.ckpt` / newest checkpoint)
- dependent job chaining with `--dependency=afterany:<jobid>`
- W&B resume (set `USE_WANDB=1` and stable run id is reused automatically in continuation mode)
