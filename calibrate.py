import torch
from transformers.generation import GenerationConfig
from tqdm import tqdm
from scipy.stats import binom
import jiwer
from conformal_utils import get_lhat

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def calibrate(model, processor, data_loader, wer_target=0.2, epsilon=0.0001, alpha=0.2, delta=0.1, num_beams=5, max_sentences=5):

    calib_loss_table = torch.Tensor([])
    for inputs in tqdm(data_loader):
        # Step 1: Predict a set of sentences for each audio file to obtain a set of sentences and their corresponding scores
        wav, labels = inputs
        data = processor(
            wav, sampling_rate=16000, return_tensors="pt"
            ).input_features
        data = data.to(device)

        gen_output = model.generate(data, num_return_sequences=num_beams, num_beams=num_beams, output_scores=True, return_dict_in_generate=True)
        gen_sequences, gen_scores, gen_beam_indices = gen_output.sequences, gen_output.scores, gen_output.beam_indices
        transition_scores = model.compute_transition_scores(gen_sequences, gen_scores, gen_beam_indices, normalize_logits=True)
        scores = transition_scores.sum(axis = 1)

        # Step 2: Filter down to k sentences
        K = max_sentences
        sentences, scores = gen_sequences[:K], scores[:K]

        # Step 3: Apply softmax on the top-k scores
        scores = torch.nn.functional.softmax(scores, dim=0)

        # Step 4: Compute the WER array for each audio.
        # TO DO: references needs to be the labels
        decoded = processor.batch_decode(sentences, skip_special_tokens=True)
        wers = torch.Tensor([jiwer.wer(reference=labels, hypothesis=sent) for sent in decoded])

        # Get proportion of conformal set sentences that have a higher WER than the target
        calib_loss_table = torch.cat((calib_loss_table, (wers >= wer_target).float().sum().unsqueeze(0)), dim=0)


    # Step 8: Initailize array from 0 to 1 with step size of precision epsilon
    lambdas = torch.linspace(0.0, 1.0, int(1 / epsilon))

    # Step 9: Get optimal lhat
    lhat = get_lhat(calib_loss_table=calib_loss_table, lambdas=lambdas, alpha=alpha, B=1)

    return lhat


def conformal_test(model, processor, test_loader, lhat, wer_target=0.2, num_beams=5, max_sentences=5):
    loss_table = torch.Tensor([])
    conformal_set_sizes = torch.Tensor([]).to(device)

    for inputs in tqdm(test_loader):
        wav, labels = inputs
        data = processor(
            wav, sampling_rate=16000, return_tensors="pt"
            ).input_features
        data = data.to(device)
        
        gen_output = model.generate(data, num_return_sequences=num_beams, num_beams=num_beams, output_scores=True, return_dict_in_generate=True)
        gen_sequences, gen_scores, gen_beam_indices = gen_output.sequences, gen_output.scores, gen_output.beam_indices
        transition_scores = model.compute_transition_scores(gen_sequences, gen_scores, gen_beam_indices, normalize_logits=True)
        scores = transition_scores.sum(axis = 1)

        K = max_sentences
        sentences, scores = gen_sequences[:K], scores[:K]

        # Step 3: Apply softmax on the top-k scores
        scores = torch.nn.functional.softmax(scores, dim=0)

        # For all cases that pass the WER threshold, add the probability of the corresponding softmax scores together
        # To get the probability of utterances being higher than our WER threshold
        cum_scores = torch.cumsum(scores, dim=0)
        index = torch.nonzero(cum_scores >= lhat)[0] + 1
        conformal_set = sentences[:index]
        conformal_set_sizes = torch.cat((conformal_set_sizes, index.unsqueeze(0)), dim=0)
        decoded = processor.batch_decode(conformal_set, skip_special_tokens=True)
        wers = torch.Tensor([jiwer.wer(reference=labels, hypothesis=sent) for sent in decoded])
        loss_table = torch.cat((loss_table, (wers >= wer_target).float().sum().unsqueeze(0)), dim=0)

    alpha_hat = loss_table.sum() / len(test_loader.dataset)
    mean_conformal_set = conformal_set_sizes.mean()

    return alpha_hat, mean_conformal_set
