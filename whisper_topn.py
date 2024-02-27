from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Sequence
import torch
import torch.nn.functional as F
import numpy as np
import whisper
from whisper.decoding import (
    SequenceRanker, 
    TokenDecoder, 
    Inference, 
    LogitFilter, 
    DecodingOptions, 
    PyTorchInference,
    SuppressBlank, 
    SuppressTokens, 
    ApplyTimestampRules, 
    DecodingResult, 
    MaximumLikelihoodRanker,
)
from whisper.audio import CHUNK_LENGTH
from whisper.tokenizer import Tokenizer, get_tokenizer
from whisper.utils import compression_ratio

# Whisper custom code to generate the top-N hypotheses
# Implementation from https://github.com/openai/whisper/discussions/1619

@dataclass(frozen=True)
class CustomDecodingResult:
    audio_features: torch.torch.Tensor
    language: str
    language_probs: Optional[Dict[str, float]] = None
    tokens: List[int] = field(default_factory=list)
    token_probs: List[float] = field(default_factory=list)
    text: str = ""
    avg_logprob: float = np.nan
    no_speech_prob: float = np.nan
    temperature: float = np.nan
    compression_ratio: float = np.nan


class CustomTokenDecoder:
    def reset(self):
        """Initialize any stateful variables for decoding a new sequence"""

    def update(self, tokens: torch.torch.Tensor, logits: torch.torch.Tensor, sum_logprobs: torch.torch.Tensor, token_probs: torch.torch.Tensor) -> Tuple[torch.torch.Tensor, bool]:
        """Specify how to select the next token, based on the current trace and logits

        Parameters
        ----------
        tokens : torch.torch.Tensor, shape = (n_batch, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence tokens

        logits : torch.torch.Tensor, shape = (n_batch, vocab_size)
            per-token logits of the probability distribution at the current step

        sum_logprobs : torch.torch.Tensor, shape = (n_batch)
            cumulative log probabilities for each sequence

        Returns
        -------
        tokens : torch.torch.Tensor, shape = (n_batch, current_sequence_length + 1)
            the tokens, appended with the selected next token

        completed : bool
            True if all sequences has reached the end of text

        """
        raise NotImplementedError

    def finalize(
        self, tokens: torch.torch.Tensor, sum_logprobs: torch.torch.Tensor, token_probs: torch.torch.Tensor
    ) -> Tuple[Sequence[Sequence[torch.torch.Tensor]], List[List[float]]]:
        """Finalize search and return the final candidate sequences

        Parameters
        ----------
        tokens : torch.torch.Tensor, shape = (n_audio, n_group, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence

        sum_logprobs : torch.torch.Tensor, shape = (n_audio, n_group)
            cumulative log probabilities for each sequence

        Returns
        -------
        tokens : Sequence[Sequence[torch.torch.Tensor]], length = n_audio
            sequence of Tensors containing candidate token sequences, for each audio input

        sum_logprobs : List[List[float]], length = n_audio
            sequence of cumulative log probabilities corresponding to the above

        """
        raise NotImplementedError
    

class CustomGreedyDecoder(CustomTokenDecoder):
    def __init__(self, temperature: float, eot: int):
        self.temperature = temperature
        self.eot = eot

    def update(self, tokens: torch.torch.Tensor, logits: torch.torch.Tensor, sum_logprobs: torch.torch.Tensor, token_probs: torch.torch.Tensor) -> Tuple[torch.torch.Tensor, bool]:
        temperature = self.temperature
        if temperature == 0:
            next_tokens = logits.argmax(dim=-1)
        else:
            next_tokens = torch.distributions.Categorical(logits=logits / temperature).sample()

        logprobs = F.log_softmax(logits.float(), dim=-1)
        probs = torch.exp(logprobs)
        current_logprobs = logprobs[torch.arange(logprobs.shape[0]), next_tokens]
        sum_logprobs += current_logprobs * (tokens[:, -1] != self.eot)

        next_tokens[tokens[:, -1] == self.eot] = self.eot
        tokens = torch.cat([tokens, next_tokens[:, None]], dim=-1)

        current_token_probs = probs[torch.arange(probs.shape[0]), next_tokens]
        token_probs = torch.cat([token_probs, current_token_probs[:, None]], dim=-1)

        # token_logits = torch.stack([logits[k, next_tokens[k]] for k in range(next_tokens .shape[0])], dim=0)
        # or use logprobs, the log softmax of the logits
        # return it along with tokens and completed

        completed = (tokens[:, -1] == self.eot).all()
        return tokens, completed, token_probs

    def finalize(self, tokens: torch.torch.Tensor, sum_logprobs: torch.torch.Tensor, token_probs: torch.torch.Tensor):
        # make sure each sequence has at least one EOT token at the end
        tokens = F.pad(tokens, (0, 1), value=self.eot)
        token_probs = F.pad(token_probs, (0, 1), value=0) # 0 ok?
        return tokens, sum_logprobs.tolist(), token_probs.tolist()

class CustomBeamSearchDecoder(CustomTokenDecoder):
    def __init__(self, beam_size: int, eot: int, inference: Inference, patience: Optional[float] = None):
        self.beam_size = beam_size
        self.eot = eot
        self.inference = inference
        self.patience = patience or 1.0
        self.max_candidates: int = round(beam_size * self.patience)
        self.finished_sequences = None

        assert self.max_candidates > 0, f"Invalid beam size ({beam_size}) or patience ({patience})"

    def reset(self):
        self.finished_sequences = None

    def update(self, tokens: torch.torch.Tensor, logits: torch.torch.Tensor, sum_logprobs: torch.torch.Tensor) -> Tuple[torch.torch.Tensor, bool]:
        if tokens.shape[0] % self.beam_size != 0:
            raise ValueError(f"{tokens.shape}[0] % {self.beam_size} != 0")

        n_audio = tokens.shape[0] // self.beam_size
        if self.finished_sequences is None:  # for the first update
            self.finished_sequences = [{} for _ in range(n_audio)]

        logprobs = F.log_softmax(logits.float(), dim=-1)
        next_tokens, source_indices, finished_sequences = [], [], []
        for i in range(n_audio):
            scores, sources, finished = {}, {}, {}

            # STEP 1: calculate the cumulative log probabilities for possible candidates
            for j in range(self.beam_size):
                idx = i * self.beam_size + j
                prefix = tokens[idx].tolist()
                for logprob, token in zip(*logprobs[idx].topk(self.beam_size + 1)):
                    new_logprob = (sum_logprobs[idx] + logprob).item()
                    sequence = tuple(prefix + [token.item()])
                    scores[sequence] = new_logprob
                    sources[sequence] = idx

            # STEP 2: rank the candidates and keep the top beam_size sequences for each audio
            saved = 0
            for sequence in sorted(scores, key=scores.get, reverse=True):
                if sequence[-1] == self.eot:
                    finished[sequence] = scores[sequence]
                else:
                    sum_logprobs[len(next_tokens)] = scores[sequence]
                    next_tokens.append(sequence)
                    source_indices.append(sources[sequence])

                    saved += 1
                    if saved == self.beam_size:
                        break

            finished_sequences.append(finished)

        tokens = torch.tensor(next_tokens, device=tokens.device)
        self.inference.rearrange_kv_cache(source_indices)

        # add newly finished sequences to self.finished_sequences
        assert len(self.finished_sequences) == len(finished_sequences)
        for previously_finished, newly_finished in zip(self.finished_sequences, finished_sequences):
            for seq in sorted(newly_finished, key=newly_finished.get, reverse=True):
                if len(previously_finished) >= self.max_candidates:
                    break  # the candidate list is full
                previously_finished[seq] = newly_finished[seq]

        # mark as completed if all audio has enough number of samples
        completed = all(
            len(sequences) >= self.max_candidates for sequences in self.finished_sequences
        )
        return tokens, completed

    def finalize(self, preceding_tokens: torch.torch.Tensor, sum_logprobs: torch.torch.Tensor, token_probs: torch.torch.Tensor):
        # collect all finished sequences, including patience, and add unfinished ones if not enough
        sum_logprobs = sum_logprobs.cpu()
        for i, sequences in enumerate(self.finished_sequences):
            if len(sequences) < self.beam_size:  # when not enough sequences are finished
                for j in list(np.argsort(sum_logprobs[i]))[::-1]:
                    sequence = preceding_tokens[i, j].tolist() + [self.eot]
                    sequences[tuple(sequence)] = sum_logprobs[i][j].item()
                    if len(sequences) >= self.beam_size:
                        break

        tokens: List[List[torch.torch.Tensor]] = [
            [torch.tensor(seq) for seq in sequences.keys()] for sequences in self.finished_sequences
        ]
        sum_logprobs: List[List[float]] = [
            list(sequences.values()) for sequences in self.finished_sequences
        ]
        return tokens, sum_logprobs


class CustomDecodingTask:
    inference: Inference
    sequence_ranker: SequenceRanker
    decoder: TokenDecoder
    logit_filters: List[LogitFilter]

    def __init__(self, model: "Whisper", options: DecodingOptions):
        self.model = model

        language = options.language or "en"
        tokenizer = get_tokenizer(model.is_multilingual, language=language, task=options.task)
        self.tokenizer: Tokenizer = tokenizer
        self.options: DecodingOptions = self._verify_options(options)

        self.n_group: int = options.beam_size or options.best_of or 1
        self.n_ctx: int = model.dims.n_text_ctx
        self.sample_len: int = options.sample_len or model.dims.n_text_ctx // 2

        self.sot_sequence: Tuple[int] = tokenizer.sot_sequence
        if self.options.without_timestamps:
            self.sot_sequence = tokenizer.sot_sequence_including_notimestamps

        self.initial_tokens: Tuple[int] = self._get_initial_tokens()
        self.sample_begin: int = len(self.initial_tokens)
        self.sot_index: int = self.initial_tokens.index(tokenizer.sot)

        # inference: implements the forward pass through the decoder, including kv caching
        self.inference = PyTorchInference(model, len(self.initial_tokens))

        # sequence ranker: implements how to rank a group of sampled sequences
        self.sequence_ranker = CustomMaximumLikelihoodRanker(options.length_penalty)

        # decoder: implements how to select the next tokens, given the autoregressive distribution
        if options.beam_size is not None:
            self.decoder = CustomBeamSearchDecoder(
                options.beam_size, tokenizer.eot, self.inference, options.patience
            )
        else:
            self.decoder = CustomGreedyDecoder(options.temperature, tokenizer.eot)

        # logit filters: applies various rules to suppress or penalize certain tokens
        self.logit_filters = []
        if self.options.suppress_blank:
            self.logit_filters.append(SuppressBlank(self.tokenizer, self.sample_begin))
        if self.options.suppress_tokens:
            self.logit_filters.append(SuppressTokens(self._get_suppress_tokens()))
        if not options.without_timestamps:
            precision = CHUNK_LENGTH / model.dims.n_audio_ctx  # usually 0.02 seconds
            max_initial_timestamp_index = None
            if options.max_initial_timestamp:
                max_initial_timestamp_index = round(self.options.max_initial_timestamp / precision)
            self.logit_filters.append(
                ApplyTimestampRules(tokenizer, self.sample_begin, max_initial_timestamp_index)
            )

    def _verify_options(self, options: DecodingOptions) -> DecodingOptions:
        if options.beam_size is not None and options.best_of is not None:
            raise ValueError("beam_size and best_of can't be given together")
        if options.temperature == 0:
            if options.best_of is not None:
                raise ValueError("best_of with greedy sampling (T=0) is not compatible")
        if options.patience is not None and options.beam_size is None:
            raise ValueError("patience requires beam_size to be given")
        if options.length_penalty is not None and not (0 <= options.length_penalty <= 1):
            raise ValueError("length_penalty (alpha) should be a value between 0 and 1")

        return options

    def _get_initial_tokens(self) -> Tuple[int]:
        tokens = list(self.sot_sequence)
        prefix = self.options.prefix
        prompt = self.options.prompt

        if prefix:
            prefix_tokens = (
                self.tokenizer.encode(" " + prefix.strip()) if isinstance(prefix, str) else prefix
            )
            if self.sample_len is not None:
                max_prefix_len = self.n_ctx // 2 - self.sample_len
                prefix_tokens = prefix_tokens[-max_prefix_len:]
            tokens = tokens + prefix_tokens

        if prompt:
            prompt_tokens = (
                self.tokenizer.encode(" " + prompt.strip()) if isinstance(prompt, str) else prompt
            )
            tokens = [self.tokenizer.sot_prev] + prompt_tokens[-(self.n_ctx // 2 - 1) :] + tokens

        return tuple(tokens)

    def _get_suppress_tokens(self) -> Tuple[int]:
        suppress_tokens = self.options.suppress_tokens

        if isinstance(suppress_tokens, str):
            suppress_tokens = [int(t) for t in suppress_tokens.split(",")]

        if -1 in suppress_tokens:
            suppress_tokens = [t for t in suppress_tokens if t >= 0]
            suppress_tokens.extend(self.tokenizer.non_speech_tokens)
        elif suppress_tokens is None or len(suppress_tokens) == 0:
            suppress_tokens = []  # interpret empty string as an empty list
        else:
            assert isinstance(suppress_tokens, list), "suppress_tokens must be a list"

        suppress_tokens.extend(
            [self.tokenizer.sot, self.tokenizer.sot_prev, self.tokenizer.sot_lm]
        )
        if self.tokenizer.no_speech is not None:
            # no-speech probability is collected separately
            suppress_tokens.append(self.tokenizer.no_speech)

        return tuple(sorted(set(suppress_tokens)))

    def _get_audio_features(self, mel: torch.Tensor):
        if self.options.fp16:
            mel = mel.half()

        if mel.shape[-2:] == (self.model.dims.n_audio_ctx, self.model.dims.n_audio_state):
            # encoded audio features are given; skip audio encoding
            audio_features = mel
        else:
            audio_features = self.model.encoder(mel)

        if audio_features.dtype != (torch.float16 if self.options.fp16 else torch.float32):
            return TypeError(f"audio_features has an incorrect dtype: {audio_features.dtype}")

        return audio_features

    def _detect_language(self, audio_features: torch.Tensor, tokens: torch.Tensor):
        languages = [self.options.language] * audio_features.shape[0]
        lang_probs = None

        if self.options.language is None or self.options.task == "lang_id":
            lang_tokens, lang_probs = self.model.detect_language(audio_features, self.tokenizer)
            languages = [max(probs, key=probs.get) for probs in lang_probs]
            if self.options.language is None:
                tokens[:, self.sot_index + 1] = lang_tokens  # write language tokens

        return languages, lang_probs

    def _main_loop(self, audio_features: torch.Tensor, tokens: torch.Tensor):
        assert audio_features.shape[0] == tokens.shape[0]
        n_batch = tokens.shape[0]
        sum_logprobs: torch.Tensor = torch.zeros(n_batch, device=audio_features.device)
        no_speech_probs = [np.nan] * n_batch

        token_probs = torch.zeros_like(tokens).float()

        try:
            for i in range(self.sample_len):
                logits = self.inference.logits(tokens, audio_features)

                if i == 0 and self.tokenizer.no_speech is not None:  # save no_speech_probs
                    probs_at_sot = logits[:, self.sot_index].float().softmax(dim=-1)
                    no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech].tolist()

                # now we need to consider the logits at the last token only
                logits = logits[:, -1]

                # apply the logit filters, e.g. for suppressing or applying penalty to
                for logit_filter in self.logit_filters:
                    logit_filter.apply(logits, tokens)

                # expand the tokens tensor with the selected next tokens
                tokens, completed, token_probs = self.decoder.update(tokens, logits, sum_logprobs, token_probs)

                if completed or tokens.shape[-1] > self.n_ctx:
                    break
        finally:
            self.inference.cleanup_caching()

        return tokens, sum_logprobs, no_speech_probs, token_probs

    @torch.no_grad()
    def run(self, mel: torch.Tensor) -> List[CustomDecodingResult]:
        self.decoder.reset()
        tokenizer: Tokenizer = self.tokenizer
        n_audio: int = mel.shape[0]

        audio_features: torch.Tensor = self._get_audio_features(mel)  # encoder forward pass
        tokens: torch.Tensor = torch.tensor([self.initial_tokens]).repeat(n_audio, 1)

        # detect language if requested, overwriting the language token
        languages, language_probs = self._detect_language(audio_features, tokens)
        if self.options.task == "lang_id":
            return [
                CustomDecodingResult(audio_features=features, language=language, language_probs=probs)
                for features, language, probs in zip(audio_features, languages, language_probs)
            ]

        # repeat the audio & text tensors by the group size, for beam search or best-of-n sampling
        audio_features = audio_features.repeat_interleave(self.n_group, dim=0)
        tokens = tokens.repeat_interleave(self.n_group, dim=0).to(audio_features.device)

        # call the main sampling loop
        tokens, sum_logprobs, no_speech_probs, token_probs = self._main_loop(audio_features, tokens)

        # reshape the tensors to have (n_audio, n_group) as the first two dimensions
        audio_features = audio_features[:: self.n_group]
        no_speech_probs = no_speech_probs[:: self.n_group]
        assert audio_features.shape[0] == len(no_speech_probs) == n_audio

        tokens = tokens.reshape(n_audio, self.n_group, -1)
        sum_logprobs = sum_logprobs.reshape(n_audio, self.n_group)

        # get the final candidates for each group, and slice between the first sampled token and EOT
        tokens, sum_logprobs, token_probs = self.decoder.finalize(tokens, sum_logprobs, token_probs)
        tokens: List[List[torch.Tensor]] = [
            [t[self.sample_begin : (t == tokenizer.eot).nonzero()[0, 0]] for t in s] for s in tokens
        ]

        # select the top-ranked sample in each group
        selected = self.sequence_ranker.rank(tokens, sum_logprobs)
        tokens: List[List[int]] = [t[i].tolist() for i, t in zip(selected, tokens)]
        texts: List[str] = [tokenizer.decode(t).strip() for t in tokens]

        sum_logprobs: List[float] = [lp[i] for i, lp in zip(selected, sum_logprobs)]
        avg_logprobs: List[float] = [lp / (len(t) + 1) for t, lp in zip(tokens, sum_logprobs)]

        fields = (texts, languages, tokens, audio_features, avg_logprobs, no_speech_probs, token_probs)
        if len(set(map(len, fields))) != 1:
            raise RuntimeError(f"inconsistent result lengths: {list(map(len, fields))}")

        return [
            CustomDecodingResult(
                audio_features=features,
                language=language,
                tokens=tokens,
                text=text,
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
                temperature=self.options.temperature,
                compression_ratio=compression_ratio(text),
                token_probs=token_probs[-len(tokens):]
            )
            for text, language, tokens, features, avg_logprob, no_speech_prob, token_probs in zip(*fields)
        ]