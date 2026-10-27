# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Commands

### Installation and Setup
```bash
# Create conda environment
conda create --name fairseq python=3.9
conda activate fairseq

# Install fairseq-signals
pip install pip==24.0
pip install --editable ./
pip install omegaconf==2.0.5 hydra-core==1.0.4

# Install required dependencies for ECG preprocessing
pip install pandas scipy wfdb

# Build Cython components
python setup.py build_ext --inplace

# For large datasets
pip install pyarrow
```

### Model Training Commands

#### Fine-tune DeepECG-SSL
```bash
CUDA_VISIBLE_DEVICES=[device_num] fairseq-hydra-train common.fp16=true task.data=[manifest_folder] \
    model.model_path=[deepecg-ssl-path] +task.npy_dataset=true model.num_labels=[num_labels] \
    criterion._name=[loss_name] checkpoint.save_dir=[checkpoint_dir] \
    --config-dir examples/w2v_cmsc/config/finetuning/ecg_transformer --config-name diagnosis
```

#### Linear Probing
```bash
CUDA_VISIBLE_DEVICES=[device_num] fairseq-hydra-train common.fp16=true task.data=[manifest_folder] \
    model.model_path=[deepecg-ssl-path] model.linear_evaluation=true +task.npy_dataset=true \
    model.num_labels=[num_labels] criterion._name=[loss_name] checkpoint.save_dir=[checkpoint_dir] \
    --config-dir examples/w2v_cmsc/config/finetuning/ecg_transformer --config-name diagnosis
```

#### End-to-End Training
```bash
CUDA_VISIBLE_DEVICES=[device_num] fairseq-hydra-train common.fp16=true task.data=[manifest_folder] \
    model.no_pretrained_weights=true +task.npy_dataset=true model.num_labels=[num_labels] \
    criterion._name=[loss_name] checkpoint.save_dir=[checkpoint_dir] \
    --config-dir examples/w2v_cmsc/config/finetuning/ecg_transformer --config-name diagnosis
```

#### WCR v2 multi-task (EchoNext 12 labels + low LVEF + incident AF, age/sex inputs)
See `docs/WCRV2_MULTITASK_RETRAIN_PLAN.md`. Builder: `scripts/preprocess/ecg/build_wcrv2_multitask_dataset.py`;
config `diagnosis_wcrv2_multitask.yaml` (`model._name=ecg_transformer_aux_classifier`, `criterion._name=masked_bce`,
NaN label = missing, manifest line `aux_path:` for `[age_z, sex]`); launcher `scripts/train_wcrv2_multitask.sh`;
evaluation `scripts/eval_wcrv2_multitask.py`; smoke test `scripts/tests/smoke_wcrv2_multitask.py`.
On a Python 3.12 container use the Python 3.9 venv `/volume/venvs/fss39/bin/python` with
`PYTHONPATH=/volume/DeepECG-SSL-finetune python -m fairseq_cli.hydra_train ...` (omegaconf<2.1 does not load on 3.12).

### Inference
```bash
CUDA_VISIBLE_DEVICES=[device_num] fairseq-hydra-inference task.data=[manifest_folder] \
    common_eval.path=[model_to_evaluate] common_eval.results_path=[results_path] \
    task.npy_dataset=true model.num_labels=[num_labels] dataset.valid_subset=[test_file] \
    --config-dir examples/w2v_cmsc/config/finetuning/ecg_transformer --config-name eval
```

### Save Model to ONNX
```bash
CUDA_VISIBLE_DEVICES=[device_num] fairseq-hydra-save common.fp16=[use_fp16] \
    common_eval.path=[model_to_evaluate] model.num_labels=[num_labels] \
    --config-dir examples/w2v_cmsc/config/finetuning/ecg_transformer --config-name eval
```

## Architecture Overview

### Main Components

#### `/fairseq_signals/` - Core Library
- **`models/`**: Contains model implementations including:
  - `ecg_transformer.py`: ECG Transformer base architecture
  - `wav2vec2/`: Wav2Vec2-based models for ECG self-supervised learning
  - `classification/`: Classification heads for downstream tasks
  - `m3ae/`, `medvill/`: Multi-modal models for ECG-text tasks
- **`criterions/`**: Loss functions (asymmetric, binary focal, MSE, etc.)
- **`data/`**: Data loading and preprocessing utilities
  - `ecg/`: ECG-specific datasets and augmentations
  - `ecg_text/`: ECG question-answering datasets
- **`tasks/`**: Task definitions (pretraining, classification, identification, QA)
- **`trainer.py`**: Main training loop implementation
- **`optim/`**: Optimizers and learning rate schedulers

#### `/examples/` - Model Configurations
- **`w2v_cmsc/`**: DeepECG-SSL configurations (Wave2Vec + CMSC + RLM)
- **`wav2vec2/`**: Basic Wave2Vec2 ECG pretraining
- **`simclr/`**: SimCLR-based pretraining
- **`scratch/`**: From-scratch training configurations
- Each contains `config/` with YAML files for pretraining and finetuning

#### `/fairseq_cli/` - Command Line Interface
- `hydra_train.py`: Training entry point using Hydra configuration
- `hydra_inference.py`: Inference entry point
- `hydra_save.py`: Model export utilities

### Key Design Patterns

1. **Hydra Configuration System**: All training/inference parameters are managed through Hydra YAML configs
2. **Registry Pattern**: Models, criterions, tasks, and optimizers are registered via decorators
3. **Task Abstraction**: Different tasks (pretraining, classification) inherit from base `Task` class
4. **Dataset Structure**: Uses TSV manifest files to specify data paths and labels

### Data Pipeline

1. **Manifest Files**: TSV files specify data paths and shapes
   - ECG data: Expected shape (n_samples, 2500, 12)
   - Labels: Can extract specific columns via `label_indexes`
2. **Preprocessing**: ECG signals normalized to 0.00488 mV scale
3. **Augmentations**: Available for self-supervised pretraining

### Loss Functions Available
- `as`: Asymmetric loss
- `bf`: Binary focal loss  
- `mse`: Mean squared error
- `bce`: Binary cross-entropy with logits
- `mlsml`: Multi-label soft margin loss

### Important Notes
- Use `common.fp16=true` when fine-tuning DeepECG-SSL
- Dataset can be NPY (`+task.npy_dataset=true`) or DataFrame (`+task.df_dataset=true`) format
- Checkpoints saved to `outputs/[date]/[time]/checkpoints/`
- For binary classification, set `model.num_labels=1`
