import argparse
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
import yaml

import torch
from datasets import Dataset, load_dataset
from peft import (
    IA3Config,
    LoraConfig,
    PrefixTuningConfig,
    PeftConfig,
    TaskType,
    get_peft_model,
    PeftModel
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    default_data_collator
)

from huggingface_hub import login

from util import *

logging.basicConfig(level=logging.DEBUG)
os.environ["WANDB_DISABLED"]="true"


def get_model(
		model_name: str,
		task_type: str,
	    model_config_args: Optional[dict] = None,
	    quant_config_args: Optional[dict] = None) -> AutoModelForCausalLM:
	"""loads hugging face model and returns it
	
	Args:
		model_name: name of model or path to checkpoint, for now restricted to causalLLM
		model_config_args: config dictionary, such as output_attention, etc. 
		quant_config_args: the quantization configuration for BitsAndBytes

	"""
	assert task_type in ("CAUSAL_LM", "SEQ_2_SEQ_LM"), "only CAUSAL_LM and SEQ_2_SEQ_LM supported"

	if model_config_args is None:
		model_config_args={}
	if quant_config_args is None: 
		quant_config_args={}


	model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, use_cache=True, **model_config_args)
	quant_config = BitsAndBytesConfig(**quant_config_args)

	if task_type == "CAUSAL_LM":	
		model = AutoModelForCausalLM.from_pretrained(
			model_name,
			quantization_config=quant_config,
			trust_remote_code=True,
			device_map="auto",
			offload_folder="offload",
			config=model_config)

	elif task_type == "SEQ_2_SEQ_LM":
		model = AutoModelForSeq2SeqLM.from_pretrained(
			model_name,
			quantization_config=quant_config,
			trust_remote_code=True,
			device_map="auto",
			offload_folder="offload",
			config=model_config)

	return model


def load_peft_model(
	model,
	peft_config_args: dict,
	peft_type: str = "lora"):
	
	"""gets peft model from base

	Args:
		model: the base model
		peft_config_args: config dicitonary for peft
		peft_type: lora, ia3, prefix

	"""
	assert peft_type in ("lora", "ia3", "prefix"), "peft_type must be a string in (lora, ia3, prefix)"

	#TODO: add other task types
	if peft_type == "lora":
		peft_config = LoraConfig(**peft_config_args)
	elif peft_type == "ia3":
		peft_config = IA3Config(**peft_config_args)
	elif peft_type == "prefix":
		peft_config = PrefixTuningConfig(**peft_config_args)
	
	peft_model = get_peft_model(model, peft_config)
	return peft_model


def load_peft_model_from_checkpoint(
	model_name: str,
	model_path: str,
	task_type: str,
	):


	if task_type == "CAUSAL_LM":	
		model = AutoModelForCausalLM.from_pretrained(
			model_name,
			load_in_8bit=True,
			trust_remote_code=True,
			device_map="auto",
			offload_folder="offload")

	elif task_type == "SEQ_2_SEQ_LM":
		model = AutoModelForSeq2SeqLM.from_pretrained(
			model_name,
			load_in_8bit=True,
			trust_remote_code=True,
			device_map="auto",
			offload_folder="offload")

	model = PeftModel.from_pretrained(model, model_path)

	return model


def tokenize_function(
	example,
    tokenizer,
	truncation,
	max_length,
	padding,
	task_type,
	input_name,
	output_name
	):

	assert task_type in ("CAUSAL_LM", "SEQ_2_SEQ_LM"), "only CAUSAL_LM and SEQ_2_SEQ_LM supported"
	
	if task_type == "CAUSAL_LM":
		example = tokenizer(example['full_prompt'], truncation=truncation, max_length=max_length, padding=padding)

	elif task_type == "SEQ_2_SEQ_LM":
			input_encodings = tokenizer(example[input_name], truncation=truncation, max_length=max_length, padding=padding)
			example["input_ids"] = input_encodings["input_ids"]
			example["attention_mask"] = input_encodings["attention_mask"]
			with tokenizer.as_target_tokenizer():
				example["labels"] = tokenizer(example[output_name], truncation=truncation, max_length=max_length, padding=padding).input_ids

	return example

def tokenize_dataset(
	dataset: Dataset,
	tokenizer,
	max_length: Optional[int] = None,
	truncation: Optional[bool] = True,
	padding: Optional[bool] = True,
	task_type: Optional[str] = "CAUSAL_LM",
	input_name: Optional[str] = "input_ids",
	output_name: Optional[str] = "label"
	):
	"""apply tokenizer to dataset

	Args:
		dataset: the HuggingFace Dataset to be tokenized
		tokenizer: the HuggingFace Tokenizer
		max_length: the max length of sequence
		truncation: whether to truncate at max length
		padding: apply padding
	"""

	tokenized_dataset =  dataset.map(tokenize_function, fn_kwargs={'tokenizer': tokenizer, 
																	'truncation': truncation, 
																	'padding': padding, 
																	'max_length': max_length,
																	'task_type': task_type,
																	'input_name': input_name,
																	'output_name': output_name}, batched=True)
	return tokenized_dataset


def train_regular(
	model,
	tokenizer,
	trainer_config: dict,
	final_save_path: str,
	train_dataset: Dataset,
	eval_dataset: Optional[Dataset]= None,
	mlm: Optional[bool] = False) -> None:
	"""get training arguments from config dict

	Args
		model: model to be trained
		trainer_config: config dict of training arguments
		final_save_path: where to save the final model
		train_dataset: tokenized training dataset
		eval_dataset: optional evaluation set
	"""

	tokenizer.pad_token = tokenizer.eos_token
	training_args = TrainingArguments(output_dir=final_save_path,**trainer_config)
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=mlm),)

	trainer.train()

	trainer.model.save_pretrained(final_save_path)
	# tokenizer.save_pretrained(peft_model_path)


#TODO add accelerate library capacity
def train_accelerate():
	pass


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--config-path", type=str)
	args, _ = parser.parse_known_args()

	with open(args.config_path, 'r') as f:
		config = yaml.safe_load(f)

	logging.debug(f"config: {config}")

	task_type = config["task_type"]
	logging.debug(task_type)

	### no MLM for now
	config["train"]["mlm"] = False

	### get access token if given for hugging face
	if config.get("access_token"):
		login(token=config.get("access_token"))


	if config.get("peft") and config["peft"].get("peft_config"):
		config["peft"]["peft_config"]["task_type"] = task_type
	elif config.get("peft"):
		config["peft"]["peft_config"] = {}
		config["peft"]["peft_config"]["task_type"] = task_type

	# turn from string into the torch datatype
	config['model']['quant_config']['bnb_4bit_compute_dtype'] = eval(config['model']['quant_config']['bnb_4bit_compute_dtype'])

	if config['model'].get('from_peft'):
		model = load_peft_model_from_checkpoint(
				config['model']['model_name'],
				config['model']['peft_checkpoint'],
				config["task_type"])

	else:
		model = get_model(config["model"]["model_name"], task_type, config["model"]["model_config"], config["model"]["quant_config"])

		if config.get("peft"):
			model = load_peft_model(model, config["peft"]["peft_config"], config["peft"]["type"])

	tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_name"])

	trainable_params, all_params = get_trainable_parameters(model)
	logging.debug(f"trainable parameters: {trainable_params / all_params * 100}%")


	if config["data"]["data_location"] == "local":
		train_data = pd.read_csv(config["data"]["train_data"])
		logging.debug(train_data.head())
		train_dataset = Dataset.from_pandas(train_data)
		if config["data"].get("eval_data"):   
			eval_data = pd.read_csv(config["data"]["eval_data"])
			eval_dataset = Dataset.from_pandas(eval_data)

	elif config["data"]["data_location"] == "hub":
		dataset = load_dataset(config["data"]["dataset_name"])
		train_dataset = dataset["train"]
		eval_dataset = dataset["validation"]


	train_dataset_tokenized = tokenize_dataset(
		train_dataset,
		tokenizer,
		config['token']['max_length'],
		config['token']['truncation'],
		config['token']['padding'],
		task_type,
		config["data"].get("input_name", "input_ids"),
		config["data"].get("output_name", "labels"))

	if config["data"].get("eval_data"):
			eval_dataset_tokenized = tokenize_dataset(
				eval_dataset,
				tokenizer,
				config['token']['max_length'],
				config['token']['truncation'],
				config['token']['padding'],
				task_type,
				config["data"].get("input_name", "input_ids"),
				config["data"].get("output_name", "labels"))
	else:
		eval_dataset_tokenized=None


	if config["train"]["type"]=="trainer":
		train_regular(model,
					  tokenizer,
					  config["train"]["train_config"],
					  config["train"]["final_save_path"],
					  train_dataset_tokenized,
					  eval_dataset_tokenized,
					  config['train']['mlm'])
	else:
		 raise Exception("only trainer is currently supported")


if __name__ == "__main__":
	main()




