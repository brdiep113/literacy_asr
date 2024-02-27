from typing import Any, Dict, List, Tuple
import torch
from torch._tensor import Tensor
from torch.nn.modules import Module
from torch.nn import Softmax
from transformers import Seq2SeqTrainer
from transformers.generation import GenerationConfig
from scipy.stats import binom
import evaluate

NUM_BEAMS = 5

class ConformalSeq2SeqTrainer(Seq2SeqTrainer):
    """Class that replaces the evaluation dataset with calibration as in conformal learning.

    """
    def __init__(self, model, args = None, data_collator = None, train_dataset = None, eval_dataset = None, calib_dataset = None,
                 tokenizer = None, model_init = None, compute_metrics = None, callbacks = None, optimizers = (None,None),
                 preprocess_logits_for_metrics = None):
            super(ConformalSeq2SeqTrainer, self).__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer,
                                                          model_init, compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)
            self.calib_dataset = calib_dataset
            self.max_sentences = 5
            self.wer = evaluate.load("wer")
            self.wer_target = 2
            self.alpha = 0.2
            self.delta = 0.1
            self.epsilon = 0.0001

            # Adapt generation config file to generate multiple sequences from beam search
            self.gen_config = GenerationConfig.from_model_config(model.config)
            self.gen_config.num_beams = NUM_BEAMS
            self.gen_config.output_scores = True
            self.gen_config.return_dict_in_generate = True


    def calibration_loop(self,
                         dataloader, 
                         description, 
                         prediction_loss_only = None,
                         ignore_keys = None,
                         metric_key_prefix = "calib",
                         ):
            
            min_wers = []
            # Calibration loop
            for step, inputs in enumerate(dataloader):
                  # Step 1: Predict a set of sentences for each audio file to obtain a set of sentences and their corresponding scores
                  sentences, scores = self.model.generate(inputs, num_return_sequences = NUM_BEAMS, generation_config = self.gen_config)

                  # TO DO: Convert output_scores into sentence scores
                  # Step 2: Filter down to k sentences
                  K = self.max_sentences
                  sentences, scores = sentences[:K], scores[:K]
                  
                  # Step 3: Apply softmax on the top-k scores
                  scores = Softmax(scores)

                  # Step 4: Compute the WER array for each audio.

                  # TO DO: references needs to be the labels
                  wer = self.wer.compute(predictions=sentences, references=inputs)
                  min_wers.extend(wer.min(axis=1))
                  
            # Step 5: Verify that WER_target >= MeanMinWER and alpha >= alpha_min
            
            # Calculate alpha_min (proportion of datapoints where minimum WER is greater than target WER)
            alpha_min = torch.mean(min_wers >= self.wer_target)
            while self.wer_target < torch.mean(min_wers) or self.alpha < alpha_min:
                  # TO DO: How do you decide how to adjust WER_min or alpha (replace with current WER_min/alpha)
                  if self.wer_target < torch.mean(min_wers):
                        self.wer_target = torch.mean(min_wers)
                  if self.alpha < alpha_min:
                        self.alpha += 0.05
                        alpha_min = torch.mean(min_wers >= self.wer_target)
            
            # Step 6: Split dataset into calibration and test dataset (just consider the calibration dataset in this loop)
            
            # Step 7: Verify that delta is greater than Bin(n, alpha)
            n = len(dataloader)
            while self.delta < binom.cdf(self.alpha * (n + 1), n, self.alpha):
                  # TO DO: How do you decide how to adjust delta or alpha
                  delta -= 0.01
            
            # Step 8: Initailize array from 0 to 1 with step size of precision epsilon
            arr = torch.linspace(0, 1, 1 / self.epsilon)
            
            # Step 9: Use binary search to find 

            # loss function
            
            arr
          
          
    def calibrate(self, 
                  calib_dataset = None,
                  ignore_keys = None,
                  metric_key_prefix: str = "calib"):
          
          self._memory_tracker.start()
          calib_dataloader = self.get_calib_dataloader()
          # model_evaluator = DecoderUtils(self.model)

          # Perform calibration
          _, wer = model_evaluator.decode(calib_dataloader)

          # Step 1: Predict a set of sentences for each audio file to obtain a set of sentences and their corresponding scores
          self.model.generate(input_features, num_return_sequences=NUM_BEAMS, generation_config=self.gen_config)
          # Step 2: For all i in [1, n] n_X_i <- min(k, n_X_i) keep k sentences if n_X_i > k and all sentences if n < k
          # Step 3: Apply a softmax function on the top-k scores
          # Step 4: Compute the WER array for each audio
          # Step 5: Verify WER target >= MeanMinWER, if not choose a higher WER target or a smaller alpha
          # Step 6: Split the dataset into 2 subsets I calib and I test
          # Step 7: Verify del