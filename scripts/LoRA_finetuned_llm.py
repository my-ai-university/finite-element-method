# Script for fine tuning a pre-existing LLM model on data (here Finite Element Method data). The script can handle data in the form of raw data or in the form of Questions-Answers in a csv file.
# Author: Computational Physics Group @ USC Jul 2024

from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from transformers import get_scheduler, set_seed 
from transformers import BitsAndBytesConfig
import bitsandbytes as bnb
from datasets import Dataset
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
import pandas as pd
from peft import LoraConfig, get_peft_model, PeftModel
import time, datetime, os
import numpy as np
# from accelerate import Accelerator
from sklearn.model_selection import KFold

def bytes_to_giga_bytes(bytes):
    return bytes / 1024 / 1024 / 1024


def create_savemodel_dir(root_path_to_save_model, save_name):
    if save_name == False:        
        date = datetime.datetime.now()
        timestamp = (str(date.year)[-2:]+ str(date.month).rjust(2,'0')+  
                    str(date.day).rjust(2,'0') 
                    + '-' + str(date.hour).rjust(2,'0') + 
                    str(date.minute).rjust(2,'0'))
        actual_path = os.path.join(root_path_to_save_model,timestamp)
    else:
        actual_path = os.path.join(root_path_to_save_model,save_name)
    
    if not os.path.exists(actual_path):
        os.mkdir(actual_path)
    return actual_path


def tokenize_function(examples,
                      tokenizer):
        encoding = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=300, return_overflowing_tokens=True, stride=10)
        encoding['labels'] = encoding['input_ids'].copy()
        return encoding


def prep_dataloader(dataset,
                    tokenizer,
                    batch_size,
                    idx,
                    ):
    #if use_refined_transcript:
    #    tokenized_datasets = dataset.map(tokenize_function(tokenizer=tokenizer), batched=True, remove_columns=["id", "text"])
        #tokenized_datasets = dataset.map(tokenize_function, batched=True)
    #if use_QA_transcript:
    tokenized_datasets = dataset.map(tokenize_function, 
                                     batched=True, 
                                     fn_kwargs={"tokenizer":tokenizer},
                                     remove_columns=["text"])

    # print("Tokenized training dataset=\n",tokenized_datasets,"[0]=",tokenized_datasets[0])

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    train_dataloader = DataLoader(tokenized_datasets, 
                                  batch_size=batch_size, 
                                  collate_fn=data_collator,
                                  sampler=torch.utils.data.SubsetRandomSampler(idx),
                                  )
    
    return train_dataloader


def train(model, device, train_dataloader, optimizer, scheduler):
    model.train()
    print("Training")
    progress_bar = tqdm(range(len(train_dataloader)))
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "overflow_to_sample_mapping"} # comment out for accelerator
        #print("Batch Shape=", {k: v.shape for k, v in batch.items()})
        optimizer.zero_grad()
        outputs = model(**batch)
        #print("Output=", outputs)
        loss = outputs.loss
        loss.backward() # comment out for accelerator
        # accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        progress_bar.update(1)

    return


def test(model, device, test_dataloader):
    print("Testing")
    progress_bar = tqdm(range(len(test_dataloader)))
    model.eval()
    running_loss = 0 
    batch_num = 0
    with torch.no_grad():
        for batch in test_dataloader:
            batch_num += 1
            batch = {k: v.to(device) for k, v in batch.items() if k != "overflow_to_sample_mapping"} # comment out for accelerator
            outputs = model(**batch)
            loss = outputs.loss
            #print("loss=", loss)
            running_loss += loss.item()   
            progress_bar.update(1)   
    epoch_loss = running_loss/batch_num
    return epoch_loss


def main(modelname = "meta-llama/Meta-Llama-3.1-8B",
         use_8bit = True,
         use_4bit = False,
         data_file = "../data/FEM_QA_Combined.csv",
         # use_refined_transcript = True,
         # use_QA_transcript = False,
         do_inference = False,
         load_saved_model = False,
         saved_model_path = "",
         num_epochs = 1,#500,
         save_path = "/project/garikipa_1359/projects/ai_ta/saved_models/",
         save_name = False,
         qlora_rank = 64,
         batch_size = 4,
         lora_dropout=0.05,
         k_folds = 5,
         debug_mode = False,
         ):
    
    out_dict = {"batch_size":batch_size,
                "num_epochs":num_epochs,
                "qlora_rank":qlora_rank,
                "lora_dropout":lora_dropout}

    # accelerator = Accelerator()

    full_path_to_save_model = create_savemodel_dir(save_path, save_name)
    set_seed(2)

    # comment these lines out if using accelerate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print("Initial memory allocated=",bytes_to_giga_bytes(torch.cuda.max_memory_allocated()))

    model_kwargs = {"device_map":"auto"}

    if use_8bit == True:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        model_kwargs["bnb_config"] = bnb_config
    elif use_4bit == True:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_quant_storage=torch.bfloat16,
            bnb_4bit_use_double_quant=True, #Saves additional memory
            )
        model_kwargs["bnb_config"] = bnb_config
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(modelname, 
                                              padding_side="left", 
                                              return_tensors="pt",
                                              )

    if not load_saved_model:
        model = AutoModelForCausalLM.from_pretrained(modelname, **model_kwargs)
        lora_config = LoraConfig(bias="none", 
                                 # init_lora_weights=False,  # not recommended
                                 use_rslora=True, 
                                 lora_alpha=16, 
                                 lora_dropout=lora_dropout,
                                 peft_type="LORA",
                                 r=qlora_rank,
                                 target_modules=["q_proj", "v_proj"],
                                 task_type="CAUSAL_LM",
                                 )
        model = get_peft_model(model, lora_config)
        print("Model loaded")
        print("Final memory allocated=",bytes_to_giga_bytes(torch.cuda.max_memory_allocated()))


    if load_saved_model:
        # need to check
        base_model = AutoModelForCausalLM.from_pretrained(modelname, **model_kwargs)
        peft_model_id = saved_model_path
        model = PeftModel.from_pretrained(base_model, peft_model_id)
        model.merge_adapter()
        print("Successfully loaded the model into memory")
        print("Final memory allocated=",bytes_to_giga_bytes(torch.cuda.max_memory_allocated()))


    model.print_trainable_parameters()
    tokenizer.pad_token = tokenizer.eos_token

    # if use_refined_transcript:
    #     data_df = pd.read_csv("../data/FEM_Garikipati_Transcripts_aggregated.csv", dtype={'text': 'string'})
    #     data_df = data_df.set_axis(['id', 'text'], axis=1)
    #     #index_list = list(range(20, 164))
    #     #data_df.drop(data_df.index[index_list], inplace=True)
        
    #if use_QA_transcript:
    data_df = pd.read_csv(data_file, dtype={'Questions': 'string', 'Answers': 'string'})
    data_df['text'] = "Question: " + data_df['Questions'] + " Answer: " + data_df['Answers'].astype("string")
    data_df = data_df[['text']]
    #index_list = list(range(50, 8203))
    #data_df.drop(data_df.index[index_list], inplace=True)

    dataset = Dataset.from_pandas(data_df)

    kfold = KFold(n_splits=k_folds, shuffle=True)
    test_loss_list = []
    for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
        if debug_mode == True:
            if fold>0:
                break
        print(f"Fold {fold+1}")
        train_dataloader = prep_dataloader(dataset, tokenizer, batch_size, train_idx)
        test_dataloader = prep_dataloader(dataset, tokenizer, batch_size, test_idx)

        for batch in train_dataloader:
            break
        print("\n Input batch shape:",{k: v.shape for k, v in batch.items()},"\n")

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        # optimizer = bnb.optim.Adam8bit(model.parameters(), lr=5e-5) # https://huggingface.co/docs/bitsandbytes/main/en/optimizers
        
        num_training_steps = num_epochs * len(train_dataloader)
        print("\n num_training_steps =", num_training_steps,"\n")
        scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=3,
            num_training_steps=num_training_steps
        )

        # train_dataloader, model, optimizer, scheduler = accelerator.prepare(train_dataloader,
        #                                                          model,
        #                                                          optimizer,
        #                                                          scheduler,
        #                                                          )

        #progress_bar = tqdm(range(num_training_steps))

        print("Prior training memory allocated=",bytes_to_giga_bytes(torch.cuda.max_memory_allocated()))

        start_time = time.time()
        model.to(device) # comment out for accelerator
        #model.train() # now in train()

        fold_loss_list = []
        for epoch in range(num_epochs):
            print(f"Epoch: {epoch+1}")
            train(model, device, train_dataloader, optimizer, scheduler)
            epoch_loss = test(model, device, test_dataloader)
            print(f"Avg. epoch test loss: {epoch_loss}")
            fold_loss_list.append(epoch_loss)
            if (epoch % 100) == 0:
                output_dir=f"{full_path_to_save_model}/epoch{epoch}/"
                model.save_pretrained(output_dir)
        
        end_time = time.time()
        test_loss_list.append(fold_loss_list)
    
        print(f"Training Completed in {end_time - start_time:.2f} seconds")
        print("Post training memory allocated=",bytes_to_giga_bytes(torch.cuda.max_memory_allocated()))

        # Save the model
        model.save_pretrained(f"{full_path_to_save_model}/final_model_fold_{fold+1}/")

    out_dict["avg_epoch_test_losses"] = test_loss_list

    out_dict["combined_avg_test_loss"] = float(np.mean(test_loss_list))
   
    model.eval()
    print("Answering your questions....")

    questions = [
        "Question: What is Finite Element Method? Answer:",
        "Question: Tell me about the assembly process in Finite Element Method Answer:",
        "Question: What is continuum phyics? Answer:",
        "Question: What is deal.II? Answer:"
    ]
    encoded_inputs = tokenizer(questions, padding=True, return_tensors="pt").to(device)
    decoded_input = tokenizer.batch_decode(encoded_inputs["input_ids"], skip_special_tokens=True)
    generated_ids = model.generate(**encoded_inputs, max_new_tokens=100)
    text_generation = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print("\n\n Questions answered:\n\n",text_generation,"\n\n")

    out_dict["text_gen"] = text_generation

    return out_dict


if __name__ == "__main__":
    # load a small model for testing purposes
    out_dict = main("HuggingFaceTB/SmolLM-135M",
                    use_8bit=False,
                    batch_size=32,
                    k_folds=5,
                    debug_mode = True)
    print(out_dict)