import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
from datasets import load_dataset, Audio, DatasetDict, load_from_disk
import evaluate
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration
from whisper_utils import DataCollatorSpeechSeq2SeqWithPadding
import webdataset as wds
from calibrate import calibrate, conformal_test


def construct_dataloader(dataset, set, batch_size=32, drop_last=False):
    data = dataset[set]
    batch_sampler = BatchSampler(RandomSampler(data[set]), batch_size=batch_size, drop_last=drop_last)
    dataloader = DataLoader(data, batch_sampler=batch_sampler)

    return dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Whisper Tutorial",
        description="Sample tutorial for running Whisper on Cedar and logging metrics with wandb"
    )
    parser.add_argument("-b", "--beams", metavar='N', type=int, nargs='+', help="Number of beams")
    parser.add_argument("-s", "--sentences", metavar='N', type=int, nargs='+', help="Number of sentences")

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load data
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small", local_files_only=True)
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", local_files_only=True, language="en", task="transcribe")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small", local_files_only=True, language="en", task="transcribe")

    # Get data loaders
    valid_set = (
        wds.WebDataset("/home/brdiep/scratch/processed_common_voice/en--en_dev_0.tar")
        .decode()
        .shuffle(size=1000)
        .to_tuple('mp3.pth', 'mp3.txt')
    )

    test_set = (
        wds.WebDataset("/home/brdiep/scratch/processed_common_voice/en--en_test_0.tar")
        .decode()
        .shuffle(size=1000)
        .to_tuple('mp3.pth', 'mp3.txt')
    )

    # Load models
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small", local_files_only=True)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.to(device)

    # Calibrate on calibration set
    lhat = calibrate(model=model,
                     processor=processor,
                     data_loader=valid_set,
                     wer_target=0.2,
                     epsilon=0.0001,
                     alpha=0.2,
                     delta=0.1,
                     num_beams=args.beams,
                     max_sentences=args.sentences
                     )
    
    # Evaluate empirical coverage alpha_hat and mean conformal prediction set size
    alpha_hat, mean_conformal_set = conformal_test(model=model,
                                                   processor=processor,
                                                   data_loader=test_set,
                                                   wer_target=0.2,
                                                   num_beams=args.beams,
                                                   max_sentences=args.sentences
                                                   )
    
    print(lhat)
    print(alpha_hat)
    print(mean_conformal_set)
