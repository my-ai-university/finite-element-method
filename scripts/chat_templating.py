import torch, os
import pandas as pd
from transformers import AutoTokenizer

class Conversation:
    def __init__(self, tokenizer, system: str = None):
        self.tokenizer = tokenizer
        self.message = []
        if system:
            self.message.append({"role": "system", "content": system})

    def get_prompt(self, assistant_header = True):
        prompt = '<|begin_of_text|>'
        # Include the system message if it exists
        for msg in self.message:
            role = msg['role']
            content = msg['content']
            prompt += f"<|start_header_id|>{role}<|end_header_id|>{content}<|eot_id|>"
        if assistant_header == True:
            # Append the assistant's role header to prompt for the next response
            prompt += "<|start_header_id|>assistant<|end_header_id|>"
        return prompt

    def generate(self, user_input, model, temp=0.7, max_new_tokens=2000, top_k=50, top_p=0.95):
        # Add the user's input to the conversation history
        self.message.append({"role": "user", "content": user_input})

        # Generate the prompt
        prompt = self.get_prompt()

        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
        # inputs = {k: v.to(device) for k, v in inputs.items()}
        if self.tokenizer.eos_token_id is None:
            self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids('</s>')
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print(f"EOS Token ID: {self.tokenizer.eos_token_id}")
        print(f"PAD Token ID: {self.tokenizer.pad_token_id}")
        # Generate the response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temp,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                # eos_token_id=self.tokenizer.convert_tokens_to_ids('<|eot_id|>'),
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode the generated tokens
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Extract the assistant's response
        assistant_response = self.extract_assistant_response(prompt, generated_text)

        # Append the assistant's response to the conversation
        self.message.append({'role': 'assistant', 'content': assistant_response})

        return assistant_response

    def extract_assistant_response(self, prompt, generated_text):
        # Llama will keep generating after the prompt submitted, this function will
        # extract only the LLM's generated output with no special tokens

        # Remove the prompt from the generated text
        response_text = generated_text[len(prompt):]

        # Split at the end-of-turn token
        if '<|eot_id|>' in response_text:
            assistant_response = response_text.split('<|eot_id|>')[0]
        else:
            assistant_response = response_text

        # Remove special token at the end and leading or trailing whitespaces
        assistant_response = assistant_response.replace('<|end_header_id|>', '')
        assistant_response = assistant_response.strip()

        return assistant_response
    

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
               system_message: str = ""):
    # qa_csv_path = input("absolute path to QA csv:  ")
    print("Place all csv files in one directory.\nEach CSV file must have a \"question\" and \"answer\" column.")
    qa_csv_dir_path = input("absolute path to directory containing all QA csv files:  ")
    df_qa = combine_csv(qa_csv_dir_path)
    out_q_list = []
    out_qa_list = []
    for i, row in df_qa.iterrows():
        question = row["question"]
        answer = row["answer"]
        conv = Conversation(tokenizer, system_message)
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

    tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                              padding_side="left", 
                                              return_tensors="pt",
                                              )
    system_message = 'You are an AI professor for a Finite Element Method (FEM) course. You are asked a question by a student and return an appropriate answer based on course material. Your response focuses on FEM fundamentals, theories, and applications as presented in the course. Use standard latex notation when replying with mathematical notation.'

    convert_QA(tokenizer,
               system_message)


if __name__ == "__main__":
    convert_QA_for_hyperparam_opt()
