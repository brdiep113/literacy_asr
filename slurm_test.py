import argparse
import os
from datasets import load_dataset, Audio, DatasetDict
import evaluate
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from whisper_utils import DataCollatorSpeechSeq2SeqWithPadding
import wandb

f = open("slurm_test.txt", "a")
f.write("hello, slurm world")
f.close()