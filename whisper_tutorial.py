import argparse
import os
import torch
from datasets import load_dataset, Audio, DatasetDict, load_from_disk
import evaluate
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from whisper_utils import DataCollatorSpeechSeq2SeqWithPadding
import wandb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Whisper Tutorial",
        description="Sample tutorial for running Whisper on Cedar and logging metrics with wandb"
    )
    parser.add_argument("-n", "--projectname", help="Name to log this run as with wandb")
    parser.add_argument("-l", "--language", help="Common voice language code for ASR model to train/evaluate on")
    parser.add_argument("-t", "--token", help="User Access Token")
    args = parser.parse_args()
    
    os.environ["WANDB_PROJECT"] = args.projectname
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"

    wandb.init(
        project=args.projectname,
        tags=["whisper", args.language],
        config = {
            "language": args.language
        }
    )

    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language=args.language, task="transcribe")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small", language=args.language, task="transcribe")

    # def prepare_dataset(batch):
    #     # load and resample audio data from 48 to 16kHz
    #     audio = batch["audio"]

    #     # compute log-Mel input features from input audio array 
    #     batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    #     # encode target text to label ids 
    #     batch["labels"] = tokenizer(batch["sentence"]).input_ids
    #     return batch
    
    # Load and preprocess data
    print("Loading and processing data...")
    common_voice = load_from_disk("~/scratch/brdiep/cv_audio2")
    # common_voice = DatasetDict()
    # common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", args.language, split="train+validation", token=args.token)
    # common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", args.language, split="test", token=args.token)
    # common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])
    # common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
    # common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=4)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    torch.cuda.empty_cache()
    
    print("Begin training...")
    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-small-hi",  # change to a repo name of your choice
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        warmup_steps=1,
        max_steps=1000,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=5,
        eval_steps=5,
        logging_steps=25,
        report_to="wandb",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        )
    
    
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        )
    
    trainer.train()

    wandb.finish()