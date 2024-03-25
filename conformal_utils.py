import numpy as np
import torch

def get_lhat(calib_loss_table, lambdas, alpha, B=1):
    n = calib_loss_table.shape[0]
    rhat = calib_loss_table.mean(axis=0)
    lhat_idx = max(np.argmax(((n/(n+1)) * rhat + B/(n+1) ) >= alpha) - 1, 0) # Can't be -1.
    return lambdas[lhat_idx]

def get_rhat(wers_mask, scores, lambd, device):
    # Get number of samples in calibration
    n = scores.shape[0]

    # Get cumulative scores of sentences
    cum_scores = torch.cumsum(scores, dim=1).to(device=device)

    # For each sample, find number of sentences before cumulative scores of sentences >= lambda
    num_sentences = (cum_scores >= lambd).float().argmax(dim=1).to(device=device)

    # Find all sets that necessarily include a sentence with WER above target
    # Note since if a sentence is found to be above the WER target, all sets including that
    # sentence will also necessarily include a sentence above WER target
    # This is equivalent to finding the first index where we find a sentence above WER target
    # and setting all sets after it to also be above WER target
    first_above_wer = (wers_mask.float()).argmax(dim=1).to(device)
    # sets_above_wer = torch.arange(wers_mask.size(1)).unsqueeze(0).to(device) >= first_above_wer.unsqueeze(1)

    # Find if sets containing just the number of sentences such that their cumulative score >= lambda
    # also contain a sentence above the WER target
    contains_above_wer = sets_above_wer[torch.arange(sets_above_wer.size(0)).to(device), num_sentences]

    # rhat is the mean loss across all samples
    rhat = contains_above_wer.float().sum() / n

    return rhat

def find_lhat(wers_mask, scores, lambdas, alpha, delta, device, B=1):
    """Binary search over the values of lambda to find the maximum rhat that satisfies the threshold."""
    n = wers_mask.shape[0]
    l, r = 0, len(lambdas) - 1
    threshold = ((n + 1) / n) * (alpha - np.sqrt(-np.log(delta) / 2 * n) - (B / n))
    rhat_l = get_rhat(wers_mask, scores, lambdas[l], device)
    rhat_r = get_rhat(wers_mask, scores, lambdas[r], device)

    # Since rhat is non-increasing for all values of lambda...

    # If right-endpoint is larger than threshold, all lambda values will be larger and there is no solution
    if rhat_r > threshold:
        return None
    
    # Otherwise...
    while l <= r:
        # If the left-endpoint satisfies the threshold, no smaller lambda values exist and that is the solution
        if rhat_l <= threshold:
            return rhat_l
        
        # Then check the midpoint
        mid = l + r // 2
        mid_rhat = get_rhat(wers_mask, scores, lambdas[mid], device)

        # If the midpoint is greater than the threshold, we must keep checking larger lambda values
        if mid_rhat > threshold:
            l = mid + 1

        # If the midpoint is less than the threshold, we must check smaller lambda values to find the smallest one
        if rhat_l <= threshold:
            r = mid
    
    return mid_rhat
