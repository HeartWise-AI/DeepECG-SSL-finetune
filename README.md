# Fairseq-signals

This is an adaption of the [`Fairseq-signals`](https://github.com/Jwoo5/fairseq-signals) for Heartwise.

It is highly recommended to create a dedicated conda environment before installing the following libraries
```
$ conda create --name fairseq python=3.9
$ conda activate fairseq
```

# Requirements and Installation
* [PyTorch](https://pytorch.org) version >= 1.5.0
* Python version >= 3.6, and <= 3.9
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **To install fairseq-signals** from source and develop locally:

```bash
git clone https://github.com/HeartWise-AI/fairseq-signals
cd fairseq-signals
pip install pip==24.0
pip install --editable ./
pip install omegaconf==2.0.5 hydra-core==1.0.4
```

* **To preprocess ECG datasets**: `pip install pandas scipy wfdb`
* **To build cython components**: `python setup.py build_ext --inplace`
* **For large datasets** install [PyArrow](https://arrow.apache.org/docs/python/install.html#using-pip): `pip install pyarrow`


# Getting Started
For general commands on ECG preprocessing, model pretraining, and other fairseq-signals specifics, please check [`Fairseq-signals`](https://github.com/Jwoo5/fairseq-signals).
Here we will focus on explaining how to train your own classification models either end-to-end or with initialization from DeepECG-SSL weights (both linear probing and fine-tuning), run inference with models within fairseq-signals, and save models in ONNX format so they can be used elsewhere for inference. 

To get the best of `DeepECG-SSL`, please make sure you prepreocessed ECGs as described by [Nolin-Lapalme et al.](https://www.medrxiv.org/content/10.1101/2025.03.02.25322575v1.full.pdf).
A docker with the preprocessing pipeline can be find [here](https://github.com/HeartWise-AI/DeepECG_Docker)

## 1.a: Fine-tune DeepECG-SSL
This will update all the weights (base model + classification head).
```shell script
$ CUDA_VISIBLE_DEVICES=[device_num] fairseq-hydra-train common.fp16=[use_fp16] task.data=[manifest_folder] \
    model.model_path=[deepecg-ssl-path] +task.npy_dataset=true model.num_labels=[num_labels] \
    criterion._name=[loss_name] checkpoint.save_dir=[checkpoint_dir] \
    --config-dir examples/w2v_cmsc/config/finetuning/ecg_transformer --config-name diagnosis
```
- `[device_num]` should be an integer representing the CUDA GPU number (starting at 0)
- `[use_fp16]` should be either `true` for `false`. When finetuning `DeepECG-SSL`, use `true`
- `[manifest_folder]` is a folder containing at least two manifest files `train.tsv` and `valid.tsv`. In the section `Manifest structure` we will give an example of these files.
- `[deepecg-ssl-path]` is the `.pt` corresponding to `DeepECG-SSL` base model. The `valid` set is used to early-stop the fine-tuning.
- `[+task.npy_dataset=true]` in case your training/validation data are save in a panda dataframe, replace this with `[+task.df_dataset=true]`.
- `[num_labels]` is an integer corresponding to the number of target classes in the classification. For binary classification, `[num_labels] = 1`.
- `[loss_name]` corresponds to the loss function. Possible values are `as` (asymetric loss), `bf` (binary focal loss), `mse` (mean square error loss), `bce` (binary cross entropy with logits loss), and `mlsml` (multilabel soft marginal loss).
- `[checkpoint_dir]` is the checkpoint directory subfolder name inside the folder/ `outputs/`. fairseq-signals automatically organizes folders using the date and time when they are created.

## 1.b: Linear probing with DeepECG-SSL
This will freeze the base model weights and only update classification weights
```shell script
$ CUDA_VISIBLE_DEVICES=[device_num] fairseq-hydra-train common.fp16=[use_fp16] task.data=[manifest_folder] \
    model.model_path=[deepecg-ssl-path] model.linear_evaluation=true +task.npy_dataset=[npy_dataset] \
    model.num_labels=[num_labels] criterion._name=[loss_name] checkpoint.save_dir=[checkpoint_dir] \
    --config-dir examples/w2v_cmsc/config/finetuning/ecg_transformer --config-name diagnosis
```
All parameters are the same as for fine-tuning, except `model.linear_evaluation` whose value is `true`

## 1.c: End-to-end training
In case you want to initialize all the weights (base transformer + classification head) randomly in order to perform end-to-end training, you the following.
```shell script
$ CUDA_VISIBLE_DEVICES=[device_num] fairseq-hydra-train common.fp16=[use_fp16] task.data=[manifest_folder] \
    model.no_pretrained_weights=true +task.npy_dataset=[npy_dataset] model.num_labels=[num_labels] \
    criterion._name=[loss_name] checkpoint.save_dir=[checkpoint_dir] \
    --config-dir examples/w2v_cmsc/config/finetuning/ecg_transformer --config-name diagnosis
```
All parameters are the same as for finetuning, expect `model.no_pretrained_weights=true` and `[deepecg-ssl-path]`, the base model path, which is not set, as it is not needed.

## 2: Inference of trained models
Once you have trained your model, either by fine-tuning, linear probing or end-to-end, the final checkpoint is saved in the corresponding directory (specified with `[checkpoint_dir]`)
Now you can run the inference on a given `.tsv` file that specifies your test data.
```shell script
$ CUDA_VISIBLE_DEVICES=[device_num] fairseq-hydra-inference task.data=[manifest_folder] \
    common_eval.path=[model_to_evaluate] common_eval.results_path=[results_path] \
    task.npy_dataset=true model.num_labels=[num_labels] dataset.valid_subset=[test_file] \
    --config-dir examples/w2v_cmsc/config/finetuning/ecg_transformer --config-name eval
```
- `[model_to_evaluate]` correspond to the `.pt` of the model we trained and we want to evaluate now
- `[results_path]`, usually set to the same value as `[model_to_evaluate]` correspond to the path were the inference logits will be saved
- `[test_file]` correspond to the `.tsv` we want to evaluate. Usually it is simple `test.tsv`

## 3: Saving trained models on `onnx` format
To be able to run inference on the trained models outside fairseq-signals envirnoment, we can use the following command.
```shell script
$ CUDA_VISIBLE_DEVICES=[device_num] fairseq-hydra-save common.fp16=[use_fp16] \
    common_eval.path=[model_to_evaluate] model.num_labels=[num_labels] \
    --config-dir examples/w2v_cmsc/config/finetuning/ecg_transformer --config-name eval
```
A file named `model.onnx` will be saved in the same folder as `[model_to_evaluate]`
Having this file, we can run it everywhere using the following code.

```
import onnxruntime as ort
import numpy as np


model_name = 'model.onnx'

def get_session(filename, use_gpu=False):
    if use_gpu:
        return ort.InferenceSession(filename, providers=["CUDAExecutionProvider"])
    else:
        return ort.InferenceSession(filename, providers=["CPUExecutionProvider"])
    

def run_session(session, X):
    """
    session: orn session, obtained from get_session
    X: numpy array of shape (batch_size, 12, 2500) and dtype np.float16 ideally
    """
    input = {
        session.get_inputs()[0].name: X,
    }
    output = session.run(None, input)
    return output


x = 0.00488*np.transpose(np.squeeze(X[0:15]), (0, 2, 1)).astype(np.float16)
use_gpu = False # can also be set to True
session = get_session(model_name, use_gpu)
y = run_session(session, x)
print(y)
```


## Example of manifest files
`train.tsv` for a classification task with `num_labels=2`. Note that the `#` used in the file are only for description. `.tsv` does not support comments. 
```
x_path: [path_to_ecgs] #shape expected is (n_ecgs, 2500, 12)
x_shape:(2500, 12, 1)
y_path:[path_to_labels] # expected shape is (n_ecgs, n_labels_or_more)
label_indexes:[0, 5]  #in this case we extract columns 0 and 5 to be our labels
```

## Citation
@article{NolinLapalme2026,
  author = {Nolin-Lapalme, Alexis and Sowa, Achille and Delfrate, Jacques and Tastet, Olivier and Corbin, Denis and Kulbay, Merve and Ozdemir, Derman and No{\"e}l, Marie-Jeanne and Marois-Blanchet, Fran{\c{c}}ois-Christophe and Harvey, Fran{\c{c}}ois and Sharma, Surbhi and Ansari, Minhaj and Chiu, I Min and D'souza, Valentina and Friedman, Sam F. and Chass{\'e}, Micha{\"e}l and Potter, Brian J. and Afilalo, Jonathan and Elias, Pierre Adil and Jabbour, Gilbert and Bahani, Mourad and Dub{\'e}, Marie-Pierre and Boyle, Patrick M. and Chatterjee, Neal A. and Barrios, Joshua and Tison, Geoffrey H. and Ouyang, David and Maddah, Mahnaz and Khurshid, Shaan and Cadrin-Tourigny, Julia and Tadros, Rafik and Hussin, Julie and Avram, Robert},
  title = {Foundation models for electrocardiogram interpretation: clinical implications},
  journal = {European Heart Journal},
  year = {2026},
  pages = {ehaf1119},
  doi = {10.1093/eurheartj/ehaf1119},
  URL = {https://doi.org/10.1093/eurheartj/ehaf1119},
  note = {Online ahead of print}
}
