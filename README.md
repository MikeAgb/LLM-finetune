# LLM-finetune

The goal of this project is to provide an easy package to fine-tune any LLM from HuggingFace from the command line. Configure the training from a yaml file allowing you to set the peft method, the training parameters, the quantization config, and more.

## Supported Features

* Task Type
    * Causal language model
    * Sequence to sequence language model
* Training type
  * Regular fine-tuning
  * LoRA
  * IA3
  * Prefix Tuning
* Model
  * Llama, Falcon, T5, MPT, and more
* Training parameters
  * epochs
  * batch size
  * learning rate
  * ...
* Quantization
 

## Quick Set Up

Use one of the included example yaml files, and point to it from the command line:

```
pip install -r requirements.txt
python main.py --config=configs/lora.yaml
```

## The Config File

I tried to make the config file as intuitive as possible, below is an example along with definitions where applicable

```
# Only needed for Llama family models
access_token: {ACCESS TOKEN HERE}
model:
  model_name: "meta-llama/Llama-2-7b-chat-hf"
  # see https://huggingface.co/docs/transformers/main_classes/configuration
  model_config:
  # quantization config parameters, do not include if you do not plan on using quantization
  quant_config:
    load_in_4bit: true
    bnb_4bit_quant_type: "nf4"
    bnb_4bit_use_double_quant: true
    bnb_4bit_compute_dtype: "torch.bfloat16"
# either CAUSAL_LM or SEQ_2_SEQ_LM
task_type: "CAUSAL_LM"
# only include if you are using a peft method
peft:
  #"lora", "ia3", "prefix
  type: "lora"
  peft_config:
    lora_alpha: 16
    lora_dropout: 0.05
    r: 8
    bias: "none"
    # see https://github.com/huggingface/peft/blob/main/src/peft/utils/other.py#L202 for target modules by model
    target_modules:
         - "q_proj"
         - "v_proj"
data:
  # "local" or "hub"
  data_location: "local"
  # path to train data
  train_data: "test.csv"
  # path to eval data
  eval_data:
token:
  max_length:
  truncation: false
  padding: false
train:
  # for now only trainer is supported. In the future I want to add support for custom train loops
  type: "trainer"
  # where to save model and checkpoints
  final_save_path: "./llama7B"
  # training parameter
  train_config:
    per_device_train_batch_size: 1
    learning_rate: 0.0003
    gradient_accumulation_steps: 4
    save_steps: 1
    logging_steps: 1
    max_grad_norm: 1
    num_train_epochs: 5
    warmup_ratio: 0.03
    lr_scheduler_type: "constant"
    report_to:
    disable_tqdm: true
    group_by_length: true
    fp16: true
    # bf16: True
    optim: "adamw_bnb_8bit"
```

## The Data

If using CAUSAL_LM, data needs to be in a single column csv file with column name="full_prompt". 
If using SEQ_2_SEQ_LM the data should be in a two-column csv file with column names "input_ids" and "labels"

## Todos

* Tensorboard integration
* Add GPTQ support

  
