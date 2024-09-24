#!/bin/sh
import pandas as pd
import LoRA_finetuned_llm
import os

def main():
    df = pd.read_csv("./configs.csv")
    df = df.set_index('task_id')
    print(f"hyperparameter settings:")
    print(f"{df}")

    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    study_id = str(os.environ['STUDY_ID'])
    q = int(df['qlora_rank'].loc[task_id])
    n_b = int(df['batch_size'].loc[task_id])
    save_path = "/project/garikipa_1359/projects/ai_ta/hyperparam_opt/"
    save_name = f"{task_id}_{study_id}"
    out_dict = LoRA_finetuned_llm.main(save_path = save_path,
                                       save_name = save_name,
                                       batch_size = n_b,
                                       qlora_rank = q)
    out_dict["study_id"] = study_id
    out_dict["task_id"] = task_id

    with open(f"{save_path}{save_name}/{task_id}_study_out.txt","w") as file:
        file.write(str(out_dict))


if __name__=="__main__":
    main()