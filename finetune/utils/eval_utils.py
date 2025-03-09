import glob
import os
import pandas as pd
import re
from transformers import TrainerCallback
import wandb

from finetune.utils.prompt import (
    FIXED_PROMPT, FIXED_PROMPT_BASE_MODEL_COMPLETION,
    FIXED_ELEPHANT_PROMPT, FIXED_ELEPHANT_PROMPT_BASE_MODEL_COMPLETION
)


class FixedPromptEvaluationCallback(TrainerCallback):
    def __init__(self,
                 model, tokenizer,
                 prompt=FIXED_PROMPT, reference=FIXED_PROMPT_BASE_MODEL_COMPLETION,
                 max_generation_length=256, eval_steps=20):

        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.tokenized_prompt = self.tokenizer(prompt, return_tensors="pt")
        self.reference = reference
        self.max_generation_length = max_generation_length
        self.eval_steps = eval_steps
        self.completion_table = {
            "step": [],
            "prompt": [],
            "completion": [],
            "reference": [],
        }

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps == 0:

            if state.is_world_process_zero:
                completion = self.eval_prompt()

                # For logging
                self.completion_table["step"].append(str(state.global_step))
                self.completion_table["prompt"].append(self.prompt)
                self.completion_table["completion"].append(completion)
                self.completion_table["reference"].append(self.reference)
                df = pd.DataFrame(self.completion_table)
                wandb.log({"completions": wandb.Table(dataframe=df)})

    def eval_prompt(self):
        self.model.peft_config['default'].inference_mode = True
        self.tokenized_prompt.to(self.model.device)
        outputs = self.model.generate(
            **self.tokenized_prompt,
            max_length=self.max_generation_length,
            temperature=0.01,  # Very low temperature
            top_k=1,  # Only consider the most likely token
            top_p=1.0,  # Disable nucleus sampling or set to a high value
        )
        completion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.model.peft_config['default'].inference_mode = False
        return completion


def get_latest_checkpoint(checkpoint_dir):
    # Get all checkpoint directories
    checkpoint_dirs = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))

    if not checkpoint_dirs:
        return None

    # Extract checkpoint numbers and create a dictionary mapping numbers to directories
    checkpoint_dict = {}
    for dir_path in checkpoint_dirs:
        match = re.search(r'checkpoint-(\d+)$', dir_path)
        if match:
            checkpoint_num = int(match.group(1))
            checkpoint_dict[checkpoint_num] = dir_path

    if not checkpoint_dict:
        return None

    # Find the highest checkpoint number
    latest_checkpoint_num = max(checkpoint_dict.keys())
    latest_checkpoint_dir = checkpoint_dict[latest_checkpoint_num]

    return latest_checkpoint_dir
