import torch


class Conversation:
    def __init__(self,
                 model,
                 tokenizer,
                 device,
                 system=""):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
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

    def generate(self,
                 user_input,
                 temp=0.7,
                 max_new_tokens=1024,
                 top_k=50,
                 top_p=0.95):

        # Add the user's input to the conversation history
        self.message.append({"role": "user", "content": user_input})

        # Generate the prompt
        prompt = self.get_prompt()

        # Tokenize the prompt
        inputs = self.tokenizer(prompt,
                                return_tensors="pt",
                                truncation=True,
                                max_length=2048).to(self.device)
        # inputs = {k: v.to(device) for k, v in inputs.items()}
        if self.tokenizer.eos_token_id is None:
            self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids('</s>')
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print(f"EOS Token ID: {self.tokenizer.eos_token_id}")
        print(f"PAD Token ID: {self.tokenizer.pad_token_id}")
        # Generate the response
        with torch.no_grad():
            outputs = self.model.generate(
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
