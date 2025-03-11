import os
import time
import json
import openai
import numpy as np
import pandas as pd

# Set your OpenAI API key from an environment variable.
openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# Define the input and output CSV filenames.
INPUT_CSV = "responses_compare_new_prompt.csv"
OUTPUT_CSV = "responses_compare_new_prompt_evaluated.csv"

# Load the CSV file.
df = pd.read_csv(INPUT_CSV)

def get_embedding(text):
    """
    Retrieves the embedding for a given text using the OpenAI embeddings API.
    Returns a list of floats representing the embedding vector.
    """
    response = openai.embeddings.create(input=[text], model="text-embedding-3-large")
    embedding = response.data[0].embedding
    return embedding

def cosine_similarity(vec_a, vec_b):
    """
    Compute the cosine similarity between two vectors.
    """
    a = np.array(vec_a)
    b = np.array(vec_b)

    # Add a small epsilon to avoid division by zero.
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

def qualitative_judgement_science_only(question, prof_ans, base_model, fine_tuned):
    """
    Uses the Chat API to qualitatively evaluate the model answers.
    Returns a tuple (winner, comment) where 'winner' is the best model and 'comment' explains the reasoning.
    """
    prompt = f"""
        You are an expert in physics and the Finite Element Method (FEM). Evaluate the following information:

        Question:
        {question}

        Professor's Answer:
        {prof_ans}

        Answer model 1:
        {base_model}

        Answer from model 2:
        {fine_tuned}

        You are a judge evaluating two model responses against a reference response. Your task is to assess ONLY content alignment, focusing on:
        - Key concepts and main ideas present in the reference
        - Factual accuracy compared to the reference
        - Critical details and examples mentioned compared to the reference

        Explicitly IGNORE writing style, grammar, sentence structure, word choice, and formatting.
        Use the professor's answer as guidance for what constitutes a correct and complete response.

        Then, decide which model provided the better answer.

        Output your result as a JSON object with exactly two keys:
        - "winner": the chosen model ("model 1" or "model 2" or "neither" if neither answer is correct or "both" if both answers are correct and very similar)
        - "comment": a brief explanation of your reasoning.

        Ensure that the JSON is the only content in your answer.
    """

    # Create the chat completion with a defined JSON schema for the response.
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "developer", 
                "content": "You are a judge evaluating answers based on scientific and technical correctness."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "evaluation_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "winner": {
                            "description": "The chosen model based on the evaluation. either 'model 1' or 'model 2' or 'neither' if neither answer is correct or 'both' if both answers are correct and very similar.",
                            "type": "string"
                        },
                        "comment": {
                            "description": "A brief explanation of your reasoning.",
                            "type": "string"
                        }
                    },
                    "required": ["winner", "comment"],
                    "additionalProperties": False
                }
            }
        }
    )

    content = response.choices[0].message.content

    # Parse and return the JSON result
    result = json.loads(content)
    return result.get("winner", ""), result.get("comment", "")

def qualitative_judgement_all_criteria(question, prof_ans, base_model, fine_tuned):
    """
    Uses the Chat API to qualitatively evaluate the model answers.
    Returns a tuple (winner, comment) where 'winner' is the best model and 'comment' explains the reasoning.
    """
    prompt = f"""
        You are an expert in physics and the Finite Element Method (FEM). Evaluate the following information:

        Question:
        {question}

        Professor's Answer:
        {prof_ans}

        Answer model 1:
        {base_model}

        Answer from model 2:
        {fine_tuned}

        Compare the two model answers based on:
        - Completeness
        - Accuracy
        - Scientific correctness
        - Human-like presentation
        - Helpfulness

        Use the professor's answer as guidance for what constitutes a correct and complete response.
        Then, decide which model provided the better answer.

        Output your result as a JSON object with exactly two keys:
        - "winner": the chosen model ("model 1" or "model 2" or "neither" if neither answer is correct or "both" if both answers are correct and very similar)
        - "comment": a brief explanation of your reasoning.

        Ensure that the JSON is the only content in your answer.
    """

    # Create the chat completion with a defined JSON schema for the response.
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "developer", 
                "content": "You are a judge evaluating answers."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "evaluation_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "winner": {
                            "description": "The chosen model based on the evaluation. either 'model 1' or 'model 2' or 'neither' if neither answer is correct or 'both' if both answers are correct and very similar.",
                            "type": "string"
                        },
                        "comment": {
                            "description": "A brief explanation of your reasoning.",
                            "type": "string"
                        }
                    },
                    "required": ["winner", "comment"],
                    "additionalProperties": False
                }
            }
        }
    )

    content = response.choices[0].message.content

    # Parse and return the JSON result
    result = json.loads(content)
    return result.get("winner", ""), result.get("comment", "")

# Lists to store the results.
#cosine_base_model_scores = []
#cosine_fine_tuned_scores = []
chosen_models_science_only = [''] * len(df)
evaluation_comments_science_only = [''] * len(df)
chosen_models_all_criteria = [''] * len(df)
evaluation_comments_all_criteria = [''] * len(df)

# Process each row.
for index, row in df.iterrows():
    question = row.iloc[1] # 0 indexed, but 0th column is the index, so 1st column is the question
    prof_ans_text = row.iloc[2]
    base_model_text = row.iloc[3]
    fine_tuned_text = row.iloc[4]

    print(f"Processing row {index + 1}/{len(df)}")

    # Cosine Similarity Evaluation
    #prof_embedding = get_embedding(prof_ans_text)
    #base_model_embedding = get_embedding(base_model_text)
    #fine_tuned_embedding = get_embedding(fine_tuned_text)

    #base_model_score = cosine_similarity(prof_embedding, base_model_embedding)
    #fine_tuned_score = cosine_similarity(prof_embedding, fine_tuned_embedding)

    #cosine_base_model_scores.append(base_model_score)
    #cosine_fine_tuned_scores.append(fine_tuned_score)

    # Qualitative Evaluation via Chat API
    winner_science_only, comment_science_only = qualitative_judgement_science_only(question, prof_ans_text, base_model_text, fine_tuned_text)
    chosen_models_science_only[index] = winner_science_only
    evaluation_comments_science_only[index] = comment_science_only

    winner_all_criteria, comment_all_criteria = qualitative_judgement_all_criteria(question, prof_ans_text, base_model_text, fine_tuned_text)
    chosen_models_all_criteria[index] = winner_all_criteria
    evaluation_comments_all_criteria[index] = comment_all_criteria

# Add the new columns
#df["Cosine_base_model"] = cosine_base_model_scores
#df["Cosine_Fine_Tuned"] = cosine_fine_tuned_scores
df["Chosen_Model_Science_Only"] = chosen_models_science_only
df["Chosen_Model_All_Criteria"] = chosen_models_all_criteria
df["Comments_Science_Only"] = evaluation_comments_science_only
df["Comments_All_Criteria"] = evaluation_comments_all_criteria

# Write the CSV file.
df.to_csv(OUTPUT_CSV, index=False)
print(f"Evaluation complete. Results saved to '{OUTPUT_CSV}'.")