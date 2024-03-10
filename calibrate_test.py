import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
from datasets import load_dataset, Audio, DatasetDict, load_from_disk
import evaluate
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration
from whisper_utils import DataCollatorSpeechSeq2SeqWithPadding
import wandb
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
    parser.add_argument("-n", "--projectname", help="Name to log this run as with wandb")
    parser.add_argument("-t", "--token", help="User Access Token")
    parser.add_argument("-b", "--beams", metavar='N', type=int, nargs='+', help="Number of beams")
    parser.add_argument("-s", "--sentences", metavar='N', type=int, nargs='+', help="Number of sentences")

    args = parser.parse_args()
    
    os.environ["WANDB_PROJECT"] = args.projectname
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"

    wandb.init(
        project=args.projectname,
        tags=["whisper", "en"],
        config = {
            "language": "en"
        }
    )
    
    # Load data
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="en", task="transcribe")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="en", task="transcribe")

    def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]
        
        # compute log-Mel input features from input audio array 
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        
        # encode target text to label ids 
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        
        return batch

    token = args.token
    cv = DatasetDict()
    cv["validation"] = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="validation", token=token, trust_remote_code=True, cache_dir="~/scratch/brdiep/.cache/huggingface/datasets")
    cv["test"] = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="test", token=token, trust_remote_code=True, cache_dir="~/scratch/brdiep/.cache/huggingface/datasets")
    cv = cv.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])
    cv = cv.cast_column("audio", Audio(sampling_rate=16000))
    cv = cv.map(prepare_dataset, remove_columns=cv.column_names["train"], num_proc=4)
    # Create data loaders
    # data = load_from_disk("~/scratch/brdiep/cv_en")
    calibDataLoader = construct_dataloader(cv, "validation") 
    evalDataLoader = construct_dataloader(cv, "test")

    # Load models
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # Calibrate on calibration set
    lhat = calibrate(model=model,
                     processor=processor,
                     data_loader=calibDataLoader,
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
                                                   data_loader=evalDataLoader,
                                                   wer_target=0.2,
                                                   num_beams=args.beams,
                                                   max_sentences=args.sentences
                                                   )
    
    wandb.run.summary["lhat"] = lhat
    wandb.run.summary["alpha_Hat"] = alpha_hat
    wandb.run.summary["mean_conformal_set_size"] = mean_conformal_set
