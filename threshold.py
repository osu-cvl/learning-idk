"""
Paper: Learning When to Say "I Don't Know"
arXiv Link: https://arxiv.org/abs/2209.04944
Authors: Nicholas Kashani Motlagh*, Jim Davis*, 
         Tim Anderson+, and Jeremy Gwinnup+
Affiliation: *Department of Computer Science & Engineering, Ohio State University
             +Air Force Research Laboratory, Wright-Patterson AFB
Corresponding Email: kashanimotlagh.1@osu.edu (First: Nicholas, Last: Kashani Motlagh)
Date: Sep 6, 2022
This research was supported by the U.S. Air Force Research Laboratory under Contract #GRT00054740 (Release #AFRL-2022-3339). 
"""

# Built-in imports
import argparse
from pathlib import Path

# External imports
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
import torch
from torch.utils.data import Dataset, DataLoader

# Local imports
from temperature_scaling import ModelWithTemperature

# DEFAULTS
# Output Path for thresholds
DEFAULT_THRESHOLD_PATH = "thresholds.pt"
# Temperature scaling
N_BINS = 15

# Thresholding algorithm (b_cdf, wilson, wilson_cc, clopper_pearson, agresti_coull)
THRESH_FUNC = "b_cdf"
# Delta value used in thresholding algorithm
DELTA = 0.05

# Batch Size
NUM_IN_BATCH = 32

def _parse_args():
    """
    Command-line arguments to the system.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='threshold.py')
    parser.add_argument('--threshold_path', type=str, default=DEFAULT_THRESHOLD_PATH, help='Path to save computed thresholds')    
    parser.add_argument('--data_path', type=str, default=None, help='Path to validation data (logits,targets)')
    parser.add_argument('--test_data_path', type=str, default=None, help='Path to test data (logits,targets)')    
    parser.add_argument('--synth', action=argparse.BooleanOptionalAction, help="Boolean flag indicating data is synthetic")
    parser.add_argument('--skip_ts', action=argparse.BooleanOptionalAction, help="Boolean flag indicating whether to skip temperature scaling.")
    parser.add_argument('--delta', type=float, default=DELTA, help='User-provided significance level')
    parser.add_argument('--thresh_func', type=str, default=THRESH_FUNC, help=f'Method to compute thresholds (b-cdf, wilson'\
                                                                            'wilson-cc, clopper-pearson, agresti-coull')
    return parser.parse_args()

class LogitDataset(Dataset):
    """Simple torch Dataset with examples as logits.
    """
    def __init__(self, samples, targets):
        """Instantiates the Logit Dataset.

        Args:
            samples (torch.tensor): Tensor of logits (samples x classes)
            targets (torch.tensor): Tensor of targets (samples)
        """
        self.samples = samples
        self.targets = targets
        self.num_classes = torch.unique(targets).numel()
        
    def __len__(self):
        return self.targets.numel()
    
    def __getitem__(self, idx):
        return self.samples[idx,:], self.targets[idx]
    
def load_data(data_path, synth=False):
    """Loads the data from data_path and returns tensors for logits and targets.

    Args:
        data_path (str or Path): Path to data (logits, targets)
        synth (bool, optional): Flag indicating whether data is synthetic. Defaults to False.

    Returns:
        Tuple: logits, targets
    """
    if not type(data_path) is Path:
        data_path = Path(data_path)
    data = torch.load(data_path)
    # Synthetic data has ground truth decision in last position
    if synth:
        return data[:, :-2].to(torch.float32), data[:, -2].to(torch.int64)
    else:
        return data[:, :-1].to(torch.float32), data[:, -1].to(torch.int64)
    
def load_decisions(data_path):
    """Loads the data from data_path and returns tensors for logits and targets.

    Args:
        data_path (str or Path): Path to data (logits, targets)
        synth (bool, optional): Flag indicating whether data is synthetic. Defaults to False.

    Returns:
        Tuple: logits, targets
    """
    if not type(data_path) is Path:
        data_path = Path(data_path)
    data = torch.load(data_path)
    return data[:, -1].to(torch.int64)
    
def get_logits_loader(logits, targets):
    """Generates a dataloader from a path of logits, targets.
    
    Args:
        logits (torch.tensor): Tensor of logits.
        targets (torch.tensor): Tensor of targets.

    Returns:
        DataLoader: DataLoader for data in data_path.
    """
    dataset = LogitDataset(logits, targets)
    return DataLoader(dataset, batch_size=NUM_IN_BATCH, shuffle=False)
    
def learn_temp(dataloader):
    """Learns per-class temperatures on logits.

    Args:
        dataloader (torch DataLoader): DataLoader for logits.

    Returns:
        ModelWithTemperature: A torch.nn.Module which simply temperature scales inputs (per-class).
    """
    model_ts = ModelWithTemperature(n_bins=N_BINS, strategy="grid", per_class=True)
    model_ts.set_temperature(dataloader, t_vals=list(torch.linspace(0.25, 4.0, 100)))
    model_ts.eval()
    return model_ts

def get_ts_data(model_ts, dataloader):
    """Extracts temperature scaled logits from a dataloader.

    Args:
        model_ts (ModelWithTemperature): A temperature scaled model.
        dataloader (DataLoader): A dataloader of logits.

    Returns:
        Tuple: ts_logits, targets
    """
    all_ts_logits = []
    all_targets = []
    for (inputs,targets) in dataloader:
        ts_logits = model_ts(inputs)
        all_ts_logits.append(ts_logits)
        all_targets.append(targets)
    all_ts_logits = torch.cat(all_ts_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    return all_ts_logits, all_targets
    
def wilson_cc_bound(k, n, delta=DELTA):
    """Generates a (1-delta) upper bound using the Wilson interval with continuity correction.
    This strategy is approximately equivalent to the Binomial CDF with delta area in the tail.

    Args:
        k (int): Number of sucesses
        n (int): Number of trials
        delta (float, optional): User defined significance level. Defaults to DELTA (0.05).

    Returns:
        float: The upper bound.
    """
    p = k/n
    q = 1.-p
    z = stats.norm.isf((1-delta))
    z2 = z**2   
    denom = 2*(n+z2)
    num = 2.*n*p+z2+1.+z*np.sqrt(z2+2-1./n+4*p*(n*q-1))
    bound = num/denom
    if p == 0:
        bound = 0.
    elif p == 1:
        bound = 1.
    return bound

def learn_thresholds(logits, targets, delta=DELTA, thresh_func=THRESH_FUNC):
    """Learns per-class thresholds on logits using the proposed approach in the paper. The method
    validates the reject region using thresh_func at a user-provided significance level.

    Args:       
        logits (torch.tensor): Tensor of logits.
        targets (torch.tensor): Tensor of targets.
        delta (float, optional): User defined significance level. Defaults to DELTA.
        thresh_func (str, optional): Implementation used to validate reject region. Options
        are (b_cdf, wilson, wilson_cc, clopper_pearson, agresti_coull). Defaults to THRESH_FUNC.

    Returns:
        torch.tensor: A tensor of per-class thresholds.
    """
    num_classes = torch.unique(targets).numel()
    thresholds = torch.zeros(num_classes)
    # Extract softmax scores
    sm_scores = torch.softmax(logits, dim=1)
    max_sms, preds = torch.max(sm_scores, dim=1)
    # Compute per-class thresholds
    for c in range(num_classes):
        class_idx = torch.where(preds == c)[0]
        thresholds[c] = learn_class_threshold(preds[class_idx], max_sms[class_idx], targets[class_idx], delta, thresh_func)
    return thresholds

def accuracy(is_correct):
    """Computes accuracy from a binary tensor.

    Args:
        is_correct (bool): Binary tensor of successes and failures.

    Returns:
        float: Accuracy of the trials.
    """
    return torch.sum(is_correct) / is_correct.numel()

def check_reject(preds, targets, delta, thresh_func):
    """Validate whether the reject region is viable using thresh_func at a delta significance level.

    Args:
        preds (torch.tensor): Tensor of predictions.
        targets (torch.tensor): Tensor of targets.
        delta (float, optional): User defined significance level. Defaults to DELTA.
        thresh_func (str, optional): Implementation used to validate reject region. Options
        are (b_cdf, wilson, wilson_cc, clopper_pearson, agresti_coull). Defaults to THRESH_FUNC.

    Returns:
        bool: Whether the reject region is viable using thresh_func at a user defined significance level.
    """
    is_correct = (preds == targets)
    k, n = torch.sum(is_correct), is_correct.numel()
    if thresh_func == "b_cdf":
        return stats.binom.cdf(k, n, 0.5) <= 1-delta
    elif thresh_func == "wilson_cc":
        return wilson_cc_bound(k, n, delta=delta) <= 0.5
    else:
        if thresh_func == "clopper_pearson":
            thresh_func = "beta"
        # We need a 1-delta single tail upper bound so alpha=2*(1-delta)
        _, ci_u = proportion_confint(k, n, alpha=2*(1-delta), method=thresh_func)
        return ci_u <= 0.5

def learn_class_threshold(preds, max_sms, targets, delta, thresh_func):
    """Learns a threshold for a single class using thresh_func at a user-defined significance level delta.

    Args:
        preds (torch.tensor): Tensor of predictions.
        max_sms (torch.tensor): Tensor of softmax scores corresponding to predictions
        targets (torch.tensor): Tensor of targets.
        delta (float, optional): User defined significance level. Defaults to DELTA.
        thresh_func (str, optional): Implementation used to validate reject region. Options
        are (b_cdf, wilson, wilson_cc, clopper_pearson, agresti_coull). Defaults to THRESH_FUNC.


    Returns:
        float: Threshold that optimizes select accuracy while adhering to constraint.
    """
    # Only need to check thresholds that optimize select accuracy
    incorrect_idx = torch.where(preds != targets)[0]
    possible_thresholds = torch.unique(max_sms[incorrect_idx])
    best_thresh, best_cov = 0, -1
    # Select accuracy
    best_sacc = accuracy(preds == targets)
    # Check possible thresholds
    for thresh in possible_thresholds:
        select_idx = torch.where(max_sms > thresh)[0]
        reject_idx = torch.where(max_sms <= thresh)[0]
        if select_idx.numel() > 0:
            sacc = accuracy(preds[select_idx] == targets[select_idx])
        else:
            # The select accuracy is undefined so reject all
            sacc = 1.1
        # Get coverage
        cov = select_idx.numel() / (select_idx.numel() + reject_idx.numel())
        # Check reject region
        if check_reject(preds[reject_idx], targets[reject_idx], delta, thresh_func):
            # Optimize select accuracy / coverage
            if sacc > best_sacc or (sacc == best_sacc and cov > best_cov):
                best_thresh = thresh
                best_sacc = sacc
                best_cov = cov

    return best_thresh

def sanity_check(logits, targets):
    """A quick sanity check that prints the accuracy of logits against targets. This ensures logits and targets
    were loaded in correctly.

    Args:
        logits (torch.tensor): Tensor of logits.
        targets (torch.tensor): Tensor of targets.
    """
    print(f"Base accuracy: {accuracy(torch.argmax(logits, dim=1) == targets)}")

def evaluate(logits, targets, thresholds, decisions=None):
    """Computes the select accuracy, reject accuracy, and coverage of the logits/targets using thresholds.
    If decisions are provided (in the case of equal-density synthetic data), then IDA will also be computed.

    Args:
        logits (torch.tensor): Tensor of logits.
        targets (torch.tensor): Tensor of targets.
        thresholds (torch.tensor): Tensor of per-class thresholds.
        decisions (torch.tensor, optional): Tensor of ideal decisions used to compute IDA.
        Only applies for synthetic equal-density datasets. Defaults to None.
    """
    # Get softmax scores
    sm_scores = torch.softmax(logits, dim=1)
    max_sms, preds = torch.max(sm_scores, dim=1)
    # Get tensor of corresponding thresholds for each prediction
    class_thresholds = thresholds[preds]
    select_idx = torch.where(max_sms > class_thresholds)[0]
    reject_idx = torch.where(max_sms <= class_thresholds)[0]
    # Compute select accuracy
    if select_idx.numel() > 0:
        sacc = accuracy(preds[select_idx] == targets[select_idx])
    else:
        sacc = -1
    # Compute reject accuracy
    if reject_idx.numel() > 0:
        racc = accuracy(preds[reject_idx] == targets[reject_idx])
    else:
        racc = -1
    # Compute coverage
    cov = select_idx.numel() / (select_idx.numel() + reject_idx.numel())
    print(f"Select Accuracy: {sacc * 100 :.1f}")
    print(f"Reject Accuracy: {racc * 100 :.1f}")
    print(f"Coverage: {cov * 100 :.1f}")
    # Compute IDA
    if decisions is not None:
        selected = (max_sms > class_thresholds)
        ida = accuracy(selected == decisions)
        print(f"IDA: {ida * 100 :.1f}")

def main(data_path, threshold_path=DEFAULT_THRESHOLD_PATH, synth=False, delta=DELTA, 
         skip_ts=False, thresh_func=THRESH_FUNC, test_data_path=None):
    # Load data to learn thresh
    print("Loading Data")
    logits, targets = load_data(data_path, synth=synth)
    decisions = None

    # Load data to evaluate thresholds
    test_logits, test_targets, test_decisions = None, None, None
    if test_data_path:
        test_logits, test_targets = load_data(test_data_path, synth=synth)
        
    # Get decisions if synthetic
    if synth:
        decisions = load_decisions(data_path)
        if test_data_path:
            test_decisions = load_decisions(test_data_path)
        
    # Print accuracy
    sanity_check(logits, targets)
    
    # Temperature scale data
    model_ts = None
    if not skip_ts:
        print("Temperature Scaling")
        data_loader = get_logits_loader(logits, targets)
        model_ts = learn_temp(data_loader)
        logits, targets = get_ts_data(model_ts, data_loader)
        
        # Temp scale test data
        if test_data_path:
            test_data_loader = get_logits_loader(test_logits, test_targets)
            test_logits, test_targets = get_ts_data(model_ts, test_data_loader)
        
    # Learn thresholds
    print("Learning Thresholds")
    thresholds = learn_thresholds(logits, targets, delta=delta, thresh_func=thresh_func)
    torch.save(thresholds, threshold_path)
    
    # Evaluate thresholds on validation data
    print(f"Evaluating {data_path}")
    evaluate(logits, targets, thresholds, decisions=decisions)
    
    # Evaluate on test data
    if test_data_path:
        print(f"Evaluating {test_data_path}")
        evaluate(test_logits, test_targets, thresholds, test_decisions)

    
if __name__ == '__main__':
    args = _parse_args()
    main(args.data_path, threshold_path=args.threshold_path, synth=args.synth, delta=args.delta, 
         skip_ts=args.skip_ts, thresh_func=args.thresh_func, test_data_path=args.test_data_path)