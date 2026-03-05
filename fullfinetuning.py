import os
import sys
from typing import List
import torch.nn as nn 
import fire
import torch
import transformers

from datasets import load_dataset
from accelerate import Accelerator
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers import CONFIG_MAPPING, MODEL_FOR_CAUSAL_LM_MAPPING,AutoConfig
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from torch.nn import functional as F

from peft import set_peft_model_state_dict

from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.prompter import Prompter
import warnings
warnings.filterwarnings("ignore", message="torch.utils.checkpoint: please pass in use_reentrant")



class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        kwargs["model"].save_pretrained(checkpoint_folder)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        torch.save({}, pytorch_model_path)
        return control


class LoadBestPeftModelCallback(TrainerCallback):
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        print(f"Loading best peft model from {state.best_model_checkpoint} (score: {state.best_metric}).")
        best_model_path = os.path.join(state.best_model_checkpoint, "adapter_model.bin")
        adapters_weights = torch.load(best_model_path)
        model = kwargs["model"]
        set_peft_model_state_dict(model, adapters_weights)
        return control

def train(
    base_model: str = "", 
    data_path: str = "",
    output_dir: str = "",
    batch_size: int = 128,
    micro_batch_size: int = 8,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    cutoff_len: int = 4096,
    lr_scheduler: str = "cosine",
    warmup_steps: int = 100, 
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    wandb_run_name: str = "",
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",

):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Params using prompt template {prompt_template_name}:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"lr_scheduler: {lr_scheduler}\n"
            f"warmup_steps: {warmup_steps}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    from huggingface_hub import login
    
    login(token=os.environ.get('HF_TOKEN'))


    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    
    
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype = torch.bfloat16,
        attn_implementation = "flash_attention_2", #eager
        device_map=device_map,
        max_memory= {0: "3GB", 1: "10GB", 2: "10GB", 3: "10GB"}
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    print(type(model))
    print(model)
    print("length of tokenizer:",len(tokenizer))

    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    
    #tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
    pad = tokenizer.pad_token_id
    
    #model.config.pad_token_id = tokenizer.pad_token_id 
    
    tokenizer.padding_side = "right"


    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"])
        
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"])
            
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token)
            
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # TODO: Speed up?
        return tokenized_full_prompt


    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        print("================== private dataset")
        data = load_dataset(data_path, token=True)

    if resume_from_checkpoint:
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            ) 
            resume_from_checkpoint = (
                True 
            ) 
            
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")


    
    train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = data['validation'].shuffle().map(generate_and_tokenize_prompt)
    


    train_data = train_data.remove_columns(data["train"].column_names)
    
    if val_data != None:
        val_data = val_data.remove_columns(data["validation"].column_names)

  
    
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=batch_size,
            warmup_ratio=0.06,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            remove_unused_columns=True,
            dataloader_num_workers=4,
            bf16=True,
            logging_steps=1,
            optim='lomo',#"paged_adamw_8bit",
            #max_grad_norm = 1,
            weight_decay = 0.01,
            evaluation_strategy="steps", 
            save_strategy="steps",
            eval_steps = 10,
            save_steps = 10, # oringinal: 1000
            lr_scheduler_type=lr_scheduler,
            output_dir=output_dir,
            gradient_checkpointing=True,
            load_best_model_at_end=True,
            group_by_length=group_by_length,
            report_to = "wandb",
            run_name=wandb_run_name
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        # callbacks=[SavePeftModelCallback, LoadBestPeftModelCallback], # ONLY USE LoadBestPeftModelCallback if val_set_size > 0
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    
    
    
    
    
    print('##########################################################################################')
    print('##################################### Training Start #####################################')
    print('##########################################################################################')
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    print('##########################################################################################')
    print('##################################### Saving now.... #####################################')
    print('##########################################################################################')


    
    #model.save_pretrained(output_dir)
    
    tokenizer.save_pretrained(output_dir)
    trainer.save_model()
    
    model.base_model.save_pretrained(output_dir)
    
    pytorch_model_path = os.path.join(output_dir, "pytorch_model.bin")
    
    torch.save({}, pytorch_model_path)


if __name__ == "__main__":
    torch.cuda.empty_cache() 
    fire.Fire(train)
