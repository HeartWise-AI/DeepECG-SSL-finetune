"""
Data pre-processing:
    1. filter samples to have at least `args.min_ecg_by_patient` corresponding sessions according to `patient_id`
    2. encode labels (patient id) and segments
"""

import argparse
import os
import pandas as pd
import wfdb
import functools
import pickle
import scipy.io
import scipy.signal
import numpy as np
import math
from tqdm import tqdm
from multiprocessing import Pool


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", metavar="DIR",
        help="root directory containing data files to pre-process"
    )
    parser.add_argument(
        "--x-path", type=str,
        help="file containing the .csv or .parquet to pre-process"
    )
    parser.add_argument(
        "--npy-path", type=str, 
        default=None,
        help="file containing the .npy of all ecgs. Usefull the .csv does not have single ecgs paths"
    )
    parser.add_argument(
        "--dest", type=str, metavar="DIR",
        help="output directory"
    )
    parser.add_argument(
        "--bin-size", type=int, default=1000,
        help="bin size to regroup ecgs in subfolders"
    )
    parser.add_argument(
        "--folder-name-length", type=int, default=10,
        help="folder name length: 0 will be added if needed"
    )
    parser.add_argument(
        "--leads",
        default="0,1,2,3,4,5,6,7,8,9,10,11",
        type=str,
        help="comma separated list of lead numbers. (e.g. 0,1 loads only lead I and lead II)"
        "note that the order is following: [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]"
    )
    parser.add_argument(
        "--sample-rate",
        default=250,
        type=int,
        help="if set, data must be sampled by this sampling rate to be processed"
    )
    parser.add_argument(
        "--sec", default=5, type=int,
        help="seconds to randomly crop to"
    )
    parser.add_argument(
        "--only-norm",
        default=False,
        type=bool,
        help="whether to preprocess only normal samples (normal sinus rhythms)"
    )
    parser.add_argument(
        "--validated-by-md",
        default=0,
        type=int,
        help="whether to use only validated by md ecgs"
    )
    parser.add_argument(
        "--max-ecgs",
        default=0,
        type=int,
        help="0 to take all ecgs of the dataframe(default value)"
    )
    parser.add_argument(
        "--add-label",
        default=False,
        type=bool,
        help="whether to add labels or not. in True, specify y-labels if needed"
    )
    parser.add_argument(
        "--y-labels",
        default=None,
        type=str,
        help="y-labels. semi-colon(;) separated list of labels"
    )
    parser.add_argument(
        "--min-ecg-by-patient",
        default=1,
        type=int,
        help="minimum ecgs by patient to keep in the filtered dataset"
    )
    parser.add_argument(
        "--patient-id-col",
        default="new_PatientID",
        type=str,
        help="patient id col name"
    )
    parser.add_argument(
        "--fname-col",
        default="npy_path",
        type=str,
        help="fname col name"
    )
    parser.add_argument(
        "--normalization",
        default=None,
        type=str,
        help="pickle normalization file"
    )
    parser.add_argument(
        "--scale",
        default=1.,
        type=float,
        help="scale factor. can be done with normalization (mean, std) = (0, 1/scale), but this is straightforward"
    )
    parser.add_argument(
        "--source-sample-rate",
        default=None,
        type=int,
        help="original sample rate of the data. if different from --sample-rate, "
             "signals are resampled using scipy.signal.resample"
    )
    parser.add_argument("--workers", metavar="N", default=1, type=int,
                       help="number of parallel workers")
    parser.add_argument("--seed", default=42, type=int, metavar="N", help="random seed")

    return parser

def main(args):
    dir_path = os.path.realpath(args.root)
    dest_path = os.path.realpath(args.dest)
    x_path = os.path.realpath(args.x_path)
    args.npy_path = None if args.npy_path is None else os.path.realpath(args.npy_path)

    scaler = None
    if args.normalization:
        with open(args.normalization, 'rb') as file:
            scaler = pickle.load(file)

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    leads = args.leads.replace(' ','').split(',')
    leads_to_load = [int(lead) for lead in leads]

    ext = x_path.split(".")[-1]
    if ext == "parquet":
        csv = pd.read_parquet(x_path)
    elif ext == "csv":
        csv = pd.read_csv(x_path)
    if args.validated_by_md:
        csv = csv[csv['validated by MD'] == 1] 
    if args.max_ecgs > 0:
        csv = csv.sample(args.max_ecgs, random_state=args.seed)
    
    patient_ids = csv[args.patient_id_col].to_numpy()
    fnames = csv[args.fname_col].to_numpy()
    if args.add_label:
        if 'RV1 + SV6\xa0> 11 mm' in csv.columns.tolist():
            csv.rename(columns={'RV1 + SV6\xa0> 11 mm': 'RV1 + SV6 > 11 mm'}, inplace=True)
        y_labels = ['Acute pericarditis', 'QS complex in V1-V2-V3', 'T wave inversion (anterior - V3-V4)', 'Right atrial enlargement','2nd degree AV block - mobitz 1','Left posterior fascicular block','Wolff-Parkinson-White (Pre-excitation syndrome)','Junctional rhythm','Premature ventricular complex',"rSR' in V1-V2",'Right superior axis','ST elevation (inferior - II, III, aVF)','Afib','ST elevation (anterior - V3-V4)','RV1 + SV6 > 11 mm','Sinusal','Monomorph','Delta wave','R/S ratio in V1-V2 >1','Third Degree AV Block','LV pacing','Nonspecific intraventricular conduction delay','ST depression (inferior - II, III, aVF)','Regular','Premature atrial complex','2nd degree AV block - mobitz 2','Left anterior fascicular block','Q wave (septal- V1-V2)','Prolonged QT','Left axis deviation','Left ventricular hypertrophy','ST depression (septal- V1-V2)','Supraventricular tachycardia','Atrial paced','Q wave (inferior - II, III, aVF)','no_qrs','T wave inversion (lateral -I, aVL, V5-V6)','Right bundle branch block','ST elevation (septal - V1-V2)','SV1 + RV5 or RV6 > 35 mm','Right axis deviation','RaVL > 11 mm','Polymorph','Ventricular tachycardia','QRS complex negative in III','ST depression (lateral - I, avL, V5-V6)','1st degree AV block','Lead misplacement','Q wave (posterior - V7-V9)','Atrial flutter','Ventricular paced','ST elevation (posterior - V7-V8-V9)','Ectopic atrial rhythm (< 100 BPM)','Early repolarization','Ventricular Rhythm','Irregularly irregular','Atrial tachycardia (>= 100 BPM)','R complex in V5-V6','ST elevation (lateral - I, aVL, V5-V6)','Brugada','Bi-atrial enlargement','Q wave (lateral- I, aVL, V5-V6)','ST upslopping','T wave inversion (inferior - II, III, aVF)','Regularly irregular','Bradycardia','qRS in V5-V6-I, aVL','Q wave (anterior - V3-V4)','Acute MI','ST depression (anterior - V3-V4)','Right ventricular hypertrophy','T wave inversion (septal- V1-V2)','ST downslopping','Left bundle branch block','Low voltage','U wave','Left atrial enlargement']
        labels = csv[y_labels].to_numpy()
    else:
        labels = np.full((len(csv),), None, dtype=object)

    table = dict()
    labels_dict = dict()
    for fname, patient_id, label in zip(fnames, patient_ids, labels):
        if patient_id in table:
            table[patient_id] += ',' + os.path.join(dir_path, fname)
            labels_dict[patient_id] += [label]
        else:
            table[patient_id] = os.path.join(dir_path, fname)
            labels_dict[patient_id] = [label]
    
    filtered = {k: [v, labels_dict.get(k, v)] for k, v in table.items() if len(v.split(',')) >= args.min_ecg_by_patient}

    np.random.seed(args.seed)

    pool = Pool(processes = args.workers)
        
    with tqdm(total=len(filtered), desc="processing") as pbar:
        func = functools.partial(
            preprocess,
            args,
            scaler,
            leads_to_load,
            dest_path,
        )
        pool.map(func, filtered.items())
        pbar.update(1)
    pool.close()
    pool.join()
        

def preprocess(args, scaler, leads_to_load, dest_path, pid_fnames):
    def load_npy(fname, npy_mmap):
        if npy_mmap is None:
            return np.load(fname).squeeze()
        else:
            # Rowid is the last purely-numeric underscore-delimited segment,
            # because pids may contain underscores
            basename = os.path.splitext(os.path.basename(fname))[0]
            parts = basename.split('_')
            rowid_str = next((p for p in reversed(parts) if p.isdigit()), None)
            if rowid_str is None:
                raise ValueError(
                    f"Cannot extract numeric row index from '{basename}' "
                    f"when loading consolidated NPY '{args.npy_path}'"
                )
            return npy_mmap[int(rowid_str)]

    pid, (fnames, labels) = pid_fnames
    fnames = fnames.split(',')
    # Open memmap once per patient, reuse across all their ECGs
    npy_mmap = np.lib.format.open_memmap(args.npy_path, mode='r') if args.npy_path else None
        
    for label, fname in zip(labels, fnames):
        basename = os.path.basename(fname)
        dest_folder = get_folder(dest_path, basename, args.bin_size, args.folder_name_length)
        try: 
            record = load_npy(fname, npy_mmap)
        except Exception as e:
            print(f"unable to load {fname}, so skipped")
            continue
        
        if scaler is not None:
            record = scaler.transform(record)
        record = args.scale * record.T

        # Resample if source sample rate differs from target
        if args.source_sample_rate is not None and args.source_sample_rate != args.sample_rate:
            target_len = int(record.shape[-1] * args.sample_rate / args.source_sample_rate)
            record = scipy.signal.resample(record, target_len, axis=-1).astype(record.dtype)

        if np.isnan(record).any():
            print(f"detected nan value at: {fname}, so skipped")
            continue

        length = record.shape[-1]
        #pid = int(pid) if pid.isdigit() else pid
        def savemat(pid, feats, label, matname):
            data = {}
            data['patient_id'] = pid
            data['idx'] = pid
            if label is not None:
                data['label'] = label
            data['curr_sample_rate'] = args.sample_rate
            data['feats'] = feats
            scipy.io.savemat(os.path.join(dest_folder, matname), data)
            
        for i, seg in enumerate(range(0, length, int(args.sec * args.sample_rate))):
            if seg + args.sec * args.sample_rate <= length:
                feats = record[leads_to_load, seg: int(seg + args.sec * args.sample_rate)]
                savemat(pid, feats, label, f"{basename}_{i}.mat")
        if label is not None:
            savemat(pid, record[leads_to_load], label, f"{basename}.mat")


def get_folder(root_destination_folder, basename, bin_size, folder_name_length):
    # Extract the filename from the path
    
    # Extract the PID from the filename (assuming PID is before the first '_')
    pid = basename.split('_')[0]
    
    # Check if PID is numeric
    if pid.isdigit():
        # Convert the PID to an integer
        pid_int = int(pid)
        
        # Calculate the bin range start (e.g., 314263 with bin size 1000 -> 314000)
        bin_start = (pid_int // bin_size) * bin_size
        
        # Pad bin_start with leading zeros to ensure the folder name has the desired length
        padded_bin_start = str(bin_start).zfill(folder_name_length)
        
        # Define the destination folder using only the padded bin_start
        destination_folder = os.path.join(root_destination_folder, padded_bin_start)
    else:
        # If PID is not numeric, move to "A-Z" subfolder
        destination_folder = os.path.join(root_destination_folder, "A-Z")
    
    # Create the destination folder if it does not exist
    os.makedirs(destination_folder, exist_ok=True)
    
    # Move or process the file
    # os.rename(file_path, os.path.join(destination_folder, filename))  # Example move operation
    return destination_folder


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)