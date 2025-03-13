import torch, os, sys
import pandas as pd
from transformers import AutoTokenizer
sys.path.insert(0, os.getcwd())
from finetune.utils.inference_utils import Conversation
from sklearn.model_selection import train_test_split
    

def combine_csv(csv_dir):
    """loads all csv files in directory, returns one df
    """
    csv_list = os.listdir(csv_dir)
    combined_df = pd.DataFrame()
    for csv_file in csv_list:
        df_qa = pd.read_csv(os.path.join(csv_dir,csv_file))
        df_qa = df_qa[["question","answer"]]
        df_qa["source"] = [csv_file for i in range(len(df_qa))]
        combined_df = pd.concat([combined_df,df_qa], ignore_index=True)
    return combined_df


def apply_chat_templating(df_qa, 
                          tokenizer, 
                          system_message, 
                          device, 
                          ):
    out_q_list = []
    out_qa_list = []
    for i, row in df_qa.iterrows():
        question = row["question"]
        answer = row["answer"]
        conv = Conversation(tokenizer, system_message, device)
        conv.message.append({"role": "user",
                             "content": question,
                             })
        out_q_prompt = conv.get_prompt(assistant_header = False)
        out_q_list.append(out_q_prompt)

        out_qa_prompt = conv.get_prompt()
        out_qa_prompt += f"{answer}<|eot_id|>"
        out_qa_list.append(out_qa_prompt)

    df_qa['Q with chat templ'] = out_q_list
    df_qa['QA with chat templ'] = out_qa_list

    return df_qa


def convert_QA(tokenizer: AutoTokenizer,
               system_message: str = "",
               device: str = "cpu",
               manual_chat_template: bool = False):
    # qa_csv_path = input("absolute path to QA csv:  ")
    print("Place all csv files in one directory.\nEach CSV file must have a \"question\" and \"answer\" column.")
    qa_csv_dir_path = input("absolute path to directory containing all QA csv files:  ")
    df_qa = combine_csv(qa_csv_dir_path)

    if manual_chat_template == True:
        df_qa = apply_chat_templating(df_qa, tokenizer, system_message, device)
    
    # divide into train and test
    df_qa_train, df_qa_test = train_test_split(df_qa, test_size=0.1, random_state=32, shuffle=True)

    if manual_chat_template == True:
        df_qa_train['QA with chat templ'].to_csv(f"{qa_csv_dir_path}/train_qa_with_chat_template.csv", index=False, header=False)    
        df_qa_test['QA with chat templ'].to_csv(f"{qa_csv_dir_path}/test_qa_with_chat_template.csv", index=False, header=False)
        
    df_qa_train.to_csv(f"{qa_csv_dir_path}/train_qa.csv", index=True, header=True)
    df_qa_test.to_csv(f"{qa_csv_dir_path}/test_qa.csv", index=True, header=True)        
    return

def convert_QA_for_hyperparam_opt():
    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                              padding_side="left", 
                                              return_tensors="pt",
                                              )
    
    with open("./data/prompts/inference.txt","r") as f:
        system_message = f.readlines()[0].rstrip('\n')

    convert_QA(tokenizer,
               system_message,
               device)


if __name__ == "__main__":
    convert_QA_for_hyperparam_opt()
