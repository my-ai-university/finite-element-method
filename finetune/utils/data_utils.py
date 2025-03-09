from datasets import Dataset
import os
import pandas as pd
from typing import List, Union


def get_dataset_qa(data_files: Union[str, List[str]]) -> pd.DataFrame:
    if isinstance(data_files, str):
        data_files = [data_files]
    elif not isinstance(data_files, list):
        raise TypeError("data_files should be a string or a list of strings representing file paths.")

    data_dfs = []
    for file_path in data_files:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        expected_columns = {
            'id': 'string',
            'question': 'string',
            'answer': 'string'
        }

        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
        available_columns = first_line.split(',')
        dtype = {col: expected_columns[col] for col in available_columns if col in expected_columns}

        data_df = pd.read_csv(file_path, dtype=dtype)
        if not {'question', 'answer'}.issubset(data_df.columns):
            raise ValueError(f"CSV file {file_path} must contain at least 'question' and 'answer' columns.")
        data_dfs.append(data_df)

    dataset = pd.concat(data_dfs, ignore_index=True)
    dataset = Dataset.from_pandas(dataset[['question', 'answer']])
    return dataset

def get_dataset_text(data_file):
    if isinstance(data_file, list):
        data_file = data_file[0]
    data_df = pd.read_csv(data_file, dtype={'text': 'string'})
    dataset = Dataset.from_pandas(data_df[['text']])
    return dataset

def make_conv(example, tokenizer, system_prompt):
    conv = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"<|start_header_id|>answer\nAnswer: {answer}"},
        ]
        for question, answer in zip(example["question"], example["answer"])
    ]
    return {
        "text": tokenizer.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=False)
    }
