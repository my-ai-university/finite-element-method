import torch, os, sys
import pandas as pd
from transformers import AutoTokenizer
sys.path.insert(0, os.getcwd())
from finetune.utils.inference_utils import Conversation
    

def combine_csv(csv_dir):
    """loads all csv files in directory, returns one df
    """
    csv_list = os.listdir(csv_dir)
    combined_df = pd.DataFrame()
    for csv_file in csv_list:
        df_qa = pd.read_csv(os.path.join(csv_dir,csv_file))
        df_qa = df_qa[["question","answer"]]
        combined_df = pd.concat([combined_df,df_qa], ignore_index=True)
    return combined_df


def convert_QA(tokenizer: AutoTokenizer,
               system_message: str = "",
               device: str = "cpu"):
    # qa_csv_path = input("absolute path to QA csv:  ")
    print("Place all csv files in one directory.\nEach CSV file must have a \"question\" and \"answer\" column.")
    qa_csv_dir_path = input("absolute path to directory containing all QA csv files:  ")
    df_qa = combine_csv(qa_csv_dir_path)
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

    df_out = pd.DataFrame(out_qa_list)
    df_out.to_csv(f"{qa_csv_dir_path}/qa_with_chat_template.csv", index=False, header=False)

    df_qa['Q with chat templ'] = out_q_list
    df_qa['QA with chat templ'] = out_qa_list
    df_qa.to_csv(f"{qa_csv_dir_path}/qa_all_with_chat_template.csv", index=True, header=True)

    return

def convert_QA_for_hyperparam_opt():
    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                              padding_side="left", 
                                              return_tensors="pt",
                                              )
    system_message = 'You are an AI professor for a Finite Element Method (FEM) course. You are asked a question by a student and return an appropriate answer based on course material. Your response focuses on FEM fundamentals, theories, and applications as presented in the course. Use standard latex notation when replying with mathematical notation.'

    convert_QA(tokenizer,
               system_message,
               device)


if __name__ == "__main__":
    convert_QA_for_hyperparam_opt()
