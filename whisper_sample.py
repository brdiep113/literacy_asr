from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.generation import GenerationConfig

TOKEN = "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"
# Select an audio file and read it:
print("Load data")
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", token=TOKEN)
audio_sample = ds[1]["audio"]
waveform = audio_sample["array"]
sampling_rate = audio_sample["sampling_rate"]

beam_size = 5

# Load the Whisper model in Hugging Face format:
print("Load models")
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
gen_cfg = GenerationConfig.from_model_config(model.config)
gen_cfg.num_beams = beam_size
gen_cfg.output_scores = True
gen_cfg.return_dict_in_generate = True
print(gen_cfg.return_dict_in_generate)

print("Process Audio")
# Use the model and processor to transcribe the audio:
input_features = processor(
    waveform, sampling_rate=sampling_rate, return_tensors="pt"
).input_features

print("Generate token IDs")
# Generate token ids
gen_output = model.generate(input_features, num_return_sequences=beam_size, num_beams=beam_size, output_scores=True, return_dict_in_generate=True)
print(gen_output)
print(gen_output[0])
print(gen_output[1])
# print("Decode tokens")
# Decode token ids to text
# transcription = processor.batch_decode(predicted_ids[0], skip_special_tokens=True)

# print(transcription)