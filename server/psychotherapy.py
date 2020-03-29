from typing import List, Tuple
import re
import os
import torch
import numpy as np
from allennlp.data import Instance
from allennlp.data.fields import LabelField, ArrayField
from allennlp.predictors import Predictor
from transformers import (
  RobertaForSequenceClassification,
  RobertaTokenizerFast, # requires tokenizers and RUST
  RobertaConfig,
  AdamW,
  get_linear_schedule_with_warmup)

from server.demo_model import DemoModel

class RobertaMiscPredictor(Predictor):
  """
    we implement a ``Predictor`` that wraps the HuggingFace Roberta implementation.
  """
  def __init__(self, model_directory: str, predictor_name: str, device="cuda") -> None:
    self.device = device
    self.config = RobertaConfig.from_pretrained(model_directory)
    # Load in model related information
    self._tokenizer = RobertaTokenizerFast.from_pretrained(model_directory, add_special_tokens=False)
    self._model = model = RobertaForSequenceClassification.from_pretrained(model_directory, config=self.config).to(device)
    self._model.eval()
    # Prepare optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
      {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
      },
      {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
      },
    ]
    self._optimizer = AdamW(optimizer_grouped_parameters)
    self._optimizer.load_state_dict(torch.load(os.path.join(model_directory, "optimizer.pt")))

  def json_to_labeled_instances(self, inputs: dict) -> List:
    inputs.pop("label", None) # don't do training if we're doing interpretation.
    outputs = self.predict_json(inputs)
    if "saliency_label" in inputs:
      # splitting for caching reason
      outputs["label"] = int(inputs["saliency_label"])
    else:
      outputs["label"] = np.argmax(outputs["logits"])
    return [outputs]

  def get_gradients(self, inputs: list) -> List:
    inputs = inputs[0]
    inp = {
      "input_ids": torch.LongTensor(inputs["model_input_ids"]).unsqueeze(0).to(self.device),
      "labels": torch.LongTensor([inputs["label"]]).unsqueeze(0).to(self.device)
    }
    loss = self._model(**inp)[0]
    loss.backward()
    grads = [{'grad_input_1': 
    self._model.roberta.embeddings.word_embeddings.weight.grad[
      inp["input_ids"].flatten()].detach().to("cpu").tolist()
    }]
    return grads

  def predict_json(self, inputs: dict) -> dict:
    if inputs.get("label", ""):
      self._train_online(inputs)

    input_string = self._prepare_input(inputs)
    model_inputs = self._tokenizer.encode_plus(
      input_string,
      max_length=False, 
      return_offsets_mapping=True)
    # HACK Huggingface tokenizers (Fast) are messed up with adding special tokens
    model_inputs["input_ids"] = model_inputs["input_ids"][1:-1] # removing a redundant bos and eos
    model_inputs["input_tokens"] = self._tokenizer.convert_ids_to_tokens(model_inputs["input_ids"])
    logits, probabilities = self._predict(model_inputs)
    
    # https://stackoverflow.com/questions/40872126/python-replace-non-ascii-character-in-string
    model_inputs["input_tokens"] = [re.sub(r'[^\x00-\x7f]',r' ', x) for x in model_inputs["input_tokens"]]
    
    output = {
      "logits": logits,
      "probabilities": probabilities, # 2 decimal points at max
      "model_input": input_string,
      "model_input_tokens": model_inputs["input_tokens"],
      "model_input_ids": model_inputs["input_ids"],
      "offset_mapping": model_inputs["offset_mapping"][1:-1] # First and Last are null for some reason.
    }
    if inputs.get("label", ""):
      output["lr"] = next(iter(self._optimizer.param_groups))["lr"] # we use one learning rate across model
    return output
  
  def _predict(self, inputs) -> Tuple[List, List]:
    inp = {
      "input_ids" : torch.LongTensor(inputs["input_ids"]).unsqueeze(0).to(self.device)
    }
    with torch.no_grad():
      logits = self._model(**inp)[0].flatten().to("cpu")
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    logits, probabilities = logits.tolist(), (probabilities * 100).tolist()
    return logits, [round(x, 2) for x in probabilities]
  
  def _prepare_input(self, inputs : dict) -> str:
    sentences = [s.strip() for s in inputs["sentence"].split("\n") if s.strip()]
    agent, task = inputs["agent"].lower(), inputs["task"].lower()

    input_string = ""
    for idx, sen in enumerate(sentences):
      speaker, utterance = list(map(lambda x: x.strip(), sen.split(":", 1)))
      if speaker != "therapist" and speaker != "patient":
        continue
      if idx == len(sentences)-1 and task == "categorize":
        input_string += f"</s> utterance: {utterance}</s>"
      else:
        if not input_string:
          input_string += f"<s> {speaker.lower()}: {utterance}</s>"
        else:
          input_string += f"</s> {speaker.lower()}: {utterance}</s>"
    return input_string
  
  def _train_online(self, inputs) -> None:
    input_string = self._prepare_input(inputs)
    model_inputs = self._tokenizer.encode_plus(
      input_string,
      max_length=False, 
      return_offsets_mapping=True)
    model_inputs["label"] = int(inputs["label"])
    self._train(model_inputs)

  def _train(self, inputs) -> None:
    self._model.zero_grad()
    self._model.train()
    inp = {
      "input_ids": torch.LongTensor(inputs["input_ids"]).unsqueeze(0).to(self.device),
      "labels": torch.LongTensor([inputs["label"]]).unsqueeze(0).to(self.device)
    }
    loss = self._model(**inp)[0]
    loss.backward()
    self._optimizer.step()
    self._model.eval()
    

class RobertaMiscModel(DemoModel):
  def predictor(self) -> Predictor:
    return RobertaMiscPredictor(self.archive_file, self.predictor_name)
      
    
  