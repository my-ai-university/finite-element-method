import pandas as pd
from transformers import TrainerCallback
import wandb

from finetune.utils.prompt import FIXED_PROMPT, FIXED_PROMPT_BASE_MODEL_COMPLETION


class FixedPromptEvaluationCallback(TrainerCallback):
    def __init__(self,
                 model, tokenizer,
                 prompt=FIXED_PROMPT, base_completion=FIXED_PROMPT_BASE_MODEL_COMPLETION,
                 max_generation_length=256, eval_steps=20):

        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.tokenized_prompt = self.tokenizer(prompt, return_tensors="pt")
        self.base_completion = base_completion
        self.max_generation_length = max_generation_length
        self.eval_steps = eval_steps
        self.completion_table = {
            "step": [],
            "prompt": [],
            "completion": [],
            "Base model completion": [],
        }

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps == 0:
            completion = self.eval_prompt()

            # For logging
            self.completion_table["step"].append(str(state.global_step))
            self.completion_table["prompt"].append(self.prompt)
            self.completion_table["completion"].append(completion)
            self.completion_table["Base model completion"].append(self.base_completion)
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