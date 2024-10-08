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
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
import time, datetime, os
import numpy as np
# from accelerate import Accelerator
from sklearn.model_selection import KFold


class EarlyStopping:
    # ref:
    # https://www.geeksforgeeks.org/how-to-handle-overfitting-in-pytorch-models-using-early-stopping/
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score + self.delta: # this was < (presumably b/c neg?) 
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


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


def get_model_kwargs(use_8bit, use_4bit):
    model_kwargs = {"device_map":"auto"}

    if use_8bit == True:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        model_kwargs["quantization_config"] = bnb_config
    elif use_4bit == True:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_quant_storage=torch.bfloat16,
            bnb_4bit_use_double_quant=True, #Saves additional memory
            )
        model_kwargs["quantization_config"] = bnb_config
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    return model_kwargs


def get_model(modelname,
              model_kwargs,
              load_saved_model,
              lora_dropout,
              qlora_rank,
              saved_model_path
              ):
    # comment these 4x lines out if using accelerate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print("Max memory allocated=",bytes_to_giga_bytes(torch.cuda.max_memory_allocated()))
    print("Initial memory allocated=",bytes_to_giga_bytes(torch.cuda.memory_allocated()))

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
    elif load_saved_model:
        # TBD: need to check
        peft_model_id = saved_model_path
        config = PeftConfig.from_pretrained(peft_model_id)
        base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, **model_kwargs)
        model = PeftModel.from_pretrained(base_model, peft_model_id, is_trainable=True)
        # model.merge_adapter() # can't use this if training LoRA weights again
        # model.merge_and_unload() # use this for inferencing, not training
        print("Successfully loaded the model into memory")
        print("Final memory allocated=",bytes_to_giga_bytes(torch.cuda.max_memory_allocated()))

    return model, device


def prep_dataloader(dataset,
                    tokenizer,
                    batch_size,
                    idx,
                    ):
    
    tokenized_datasets = dataset.map(tokenize_function, 
                                     batched=True, 
                                     fn_kwargs={"tokenizer":tokenizer},
                                     remove_columns=["text"])

    # print("Tokenized dataset example=\n",tokenized_datasets,"[0]=",tokenized_datasets[0])

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
    running_loss = 0 
    batch_num = 0
    for batch in train_dataloader:
        batch_num += 1
        batch = {k: v.to(device) for k, v in batch.items() if k != "overflow_to_sample_mapping"} # comment out for accelerator
        #print("Batch Shape=", {k: v.shape for k, v in batch.items()})
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        running_loss += loss.item() 
        loss.backward() # comment out for accelerator
        # accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        progress_bar.update(1)
    train_loss = running_loss/batch_num
    return train_loss


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
            running_loss += loss.item()   
            progress_bar.update(1)   
    epoch_loss = running_loss/batch_num
    return epoch_loss


def answer_questions(questions,
                     tokenizer,
                     device,
                     model):
    encoded_inputs = tokenizer(questions, padding=True, return_tensors="pt").to(device)
    decoded_input = tokenizer.batch_decode(encoded_inputs["input_ids"], skip_special_tokens=True)
    generated_ids = model.generate(**encoded_inputs, max_new_tokens=200) # max response length
    text_generation = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return text_generation

def sample_responses(N_responses,
                     train_dl,
                     test_dl,
                     tokenizer,
                     model,
                     device,
                     ):
    model.eval()
    train_qa_orig = []
    test_qa_orig = []
    train_questions = []
    test_questions = []
    while len(train_questions) <= N_responses:
        encoded_questions = next(iter(train_dl))
        decoded_questions = tokenizer.batch_decode(encoded_questions["input_ids"], skip_special_tokens=True)
        decoded_question = decoded_questions[0]
        train_qa_orig.append(decoded_question)
        decoded_question = decoded_question.split("Answer:")[0]
        train_questions.append(f"{decoded_question} Answer: ")    
    while len(test_questions) <= N_responses:
        encoded_questions = next(iter(test_dl))
        decoded_questions = tokenizer.batch_decode(encoded_questions["input_ids"], skip_special_tokens=True)
        decoded_question = decoded_questions[0]
        test_qa_orig.append(decoded_question)
        decoded_question = decoded_question.split("Answer:")[0]
        test_questions.append(f"{decoded_question} Answer: ")
    
    train_qa = answer_questions(train_questions, tokenizer, device, model)
    test_qa = answer_questions(test_questions, tokenizer, device, model)
    response_dict = {"train_qa":train_qa,
                     "train_qa_orig":train_qa_orig,
                     "test_qa":test_qa,
                     "test_qa_orig":test_qa_orig}
    return response_dict


def main(modelname = "meta-llama/Meta-Llama-3.1-8B",
         use_8bit = True,
         use_4bit = False,
         data_file = "../data/FEM_QA_Combined.csv",
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
         run_one_fold = False,
         ):
    
    out_dict = {"batch_size":batch_size,
                "num_epochs":num_epochs,
                "qlora_rank":qlora_rank,
                "lora_dropout":lora_dropout}

    # accelerator = Accelerator()

    full_path_to_save_model = create_savemodel_dir(save_path, save_name)
    set_seed(2)

    model_kwargs = get_model_kwargs(use_8bit, use_4bit)  
    tokenizer = AutoTokenizer.from_pretrained(modelname, 
                                              padding_side="left", 
                                              return_tensors="pt",
                                              )
    tokenizer.pad_token = tokenizer.eos_token
        
    data_df = pd.read_csv(data_file, dtype={'Questions': 'string', 'Answers': 'string'})
    data_df['text'] = "Question: " + data_df['Questions'] + " Answer: " + data_df['Answers'].astype("string")
    data_df = data_df[['text']]
    print(f"dataframe loaded, shape = {data_df.shape}, head = {data_df.head()}")
    dataset = Dataset.from_pandas(data_df)

    kfold = KFold(n_splits=k_folds, shuffle=True)
    train_loss_list = []
    test_loss_list = []
    for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
        if run_one_fold == True:
            if fold>0:
                break
        print("============================")
        print(f"Fold {fold+1}")

        model, device = get_model(modelname, 
                                  model_kwargs, 
                                  load_saved_model, 
                                  lora_dropout, 
                                  qlora_rank, 
                                  saved_model_path
                                  )
        model.print_trainable_parameters()

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

        print("Prior training memory allocated=",bytes_to_giga_bytes(torch.cuda.max_memory_allocated()))
        start_time = time.time()
        model.to(device) # comment out for accelerator
        fold_train_loss_list = []
        fold_test_loss_list = []
        # early_stopping = EarlyStopping(patience=5, delta=0.01) # don't use w/ k-fold CV
        for epoch in range(num_epochs):
            print("--------------")
            print(f"Epoch: {epoch+1}")
            epoch_train_loss = train(model, device, train_dataloader, optimizer, scheduler)
            epoch_test_loss = test(model, device, test_dataloader)
            print(f"Epoch {epoch+1} avg. test loss: {epoch_test_loss}")
            fold_train_loss_list.append(epoch_train_loss)
            fold_test_loss_list.append(epoch_test_loss)
            if (epoch % 100) == 0:
                output_dir=f"{full_path_to_save_model}/epoch{epoch}/"
                model.save_pretrained(output_dir)
            # early_stopping(epoch_test_loss)
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break
                
        end_time = time.time()
        train_loss_list.append(fold_train_loss_list)
        test_loss_list.append(fold_test_loss_list)
        print(f"Training Completed in {end_time - start_time:.2f} seconds")
        print("Post training memory allocated=",bytes_to_giga_bytes(torch.cuda.max_memory_allocated()))

        # Save the model
        model.save_pretrained(f"{full_path_to_save_model}/final_model_fold_{fold+1}/")

        # Save 5 random train/test responses
        response_dict = sample_responses(5,
                                         train_dataloader,
                                         test_dataloader,
                                         tokenizer,
                                         model,
                                         device)
        
        out_dict[f"fold {fold} q/a"] = response_dict

        if ((fold+1) < k_folds) and (run_one_fold==False):
            del (model,train_dataloader,test_dataloader,optimizer,scheduler)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    out_dict["avg_epoch_train_losses"] = train_loss_list
    out_dict["avg_epoch_test_losses"] = test_loss_list
    out_dict["combined_avg_test_loss"] = float(np.mean(test_loss_list,axis=0)[-1])
   
    # this is just evaluating for the final fold. need to move up and change if we want to save all
    model.eval()
    print("=====================")
    print("Answering your questions....")
    questions = [
        "Question: What is Finite Element Method? Answer:",
        "Question: Tell me about the assembly process in Finite Element Method Answer:",
        "Question: What is continuum phyics? Answer:",
        "Question: What is deal.II? Answer:"
    ]
    text_generation = answer_questions(questions, tokenizer, device, model)
    print("\n\n Questions answered:\n\n",text_generation,"\n\n")

    out_dict["text_gen"] = text_generation

    return out_dict


if __name__ == "__main__":
    if True:
        # load a small model for testing purposes
        out_dict = main("HuggingFaceTB/SmolLM-135M",
                        num_epochs = 1,
                        use_8bit=False,
                        batch_size=32,
                        k_folds=5,
                        run_one_fold = True,
                        )
    if False:
        # test loading saved peft model
        out_dict = main(num_epochs = 1,
                        use_8bit=True,
                        load_saved_model = True,
                        save_path = "/project/garikipa_1359/projects/ai_ta/hyperparam_opt/",
                        save_name = "testing",
                        saved_model_path = "/project/garikipa_1359/projects/ai_ta/hyperparam_opt/archive/26047541_1/final_model_fold_1",
                        batch_size=4,
                        k_folds=5,
                        run_one_fold = True,
                        )
        
    print(out_dict)