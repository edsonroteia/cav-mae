Pretraining Scripts
- `run_cavmae_pretrain_scale++.sh`: Pretrain CAV-MAE Scale++ with 256 batch size. Requires largest GPUs (4x48GB).
- `run_cavmae_pretrain_scale+.sh`: Pretrain CAV-MAE Scale+ with 120 batch size. Requires larger GPUs (4x24GB).
- `run_cavmae_pretrain_base.sh`: Pretrain CAV-MAE with 48 batch size. Requires smaller GPUs (4x12GB).
- `run_cavjepa_pretrain.sh`: Baseline CAV-JEPA pretraining (MAE init, 25 epochs).
- `run_cavjepa_ablations.sh <ID>`: CAV-JEPA ablation/sweep launcher with fail-fast behavior.
- `launch_cavjepa_priority_sweeps.sh [ID ...]`: Submit the prioritized JEPA sweeps.

Finetuning Scrips
- `run_cavmae_ft_bal.sh`: Finetune CAV-MAE Scale++ on balanced AS-20K.
- `run_cavmae_ft_bal_audioonly.sh`: Finetune CAV-MAE Scale++ on balanced AS-20K with audio only.
- `run_cavmae_ft_bal_videoonly.sh`: Finetune CAV-MAE Scale++ on balanced AS-20K with visual data only.
- `run_cavmae_ft_full.sh`: Finetune CAV-MAE Scale++ on full AS-2M. 

Retrieval Evaluation
- `run_retrieval_scalep.sh`: Evaluate original CAV-MAE Scale+ retrieval.
- `run_retrieval_scalepp.sh`: Evaluate original CAV-MAE Scale++ retrieval.
- `run_retrieval_cavjepa_checkpoints.sh [EXP_DIR] [START] [END] [STEP]`: Evaluate JEPA checkpoints every N epochs and write a summary CSV.

Useful Examples
- Submit the full JEPA priority set:
  `./launch_cavjepa_priority_sweeps.sh`
- Submit only contrastive-weight sweeps:
  `./launch_cavjepa_priority_sweeps.sh A1 A2a A2b A2c A2d`
- Evaluate JEPA checkpoints at epochs 5,10,15,20,25:
  `sbatch run_retrieval_cavjepa_checkpoints.sh /weka/kuehne/kqr867/code/cav-mae/egs/audioset/exp/cavjepa-audioset-lr1e-4-epoch25-bs120-mr0.75-mom0.996-0.999-pred4 5 25 5`
