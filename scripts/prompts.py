from typing import List
from pydantic import BaseModel
from openai import OpenAI
import json

class Qs(BaseModel):
    question: str
    coverage: int

class QResponse(BaseModel):
    all_questions: List[Qs]

def gen_questions(client: OpenAI, text: str, k: int, model: str = "gpt-4o-mini") -> QResponse:
    
    system_prompt = f"""
    You are an AI assistant specialized in creating educational content for Finite Element Method (FEM).
    Generate comprehensive set of questions on topics related to FEM from the input text. **Only questions, no answer is needed.** Follow these guidelines:

    1. Questions:
    - Focus on fundamental concepts, theories, and general applications of FEM.
    - Ensure that the questions are relevant to the input text, and can be at least partially answered using the provided text.
    - Emphasize broad understanding rather than niche knowledge.
    - Questions can be of any length needed to fully express the concept being tested.
    - Complex questions involving multiple parts or mathematical derivations are encouraged.

    2. Mathematical Notation:
    - Use LaTeX formatting for mathematical expressions
    - For inline equations, use single $ wrapper (e.g., "Calculate the strain energy $U = \\frac{1}{2}\\int_V \\sigma\\epsilon dV$")
    - For display equations, use double $$ wrapper, e.g.:
        "Derive the stiffness matrix given the following stress-strain relationship:
        $$
        \\begin{{bmatrix}}
        \\sigma_{{xx}} \\\\ \\sigma_{{yy}} \\\\ \\tau_{{xy}}
        \\end{{bmatrix}} =
        \\begin{{bmatrix}}
        D_{{11}} & D_{{12}} & 0 \\\\
        D_{{21}} & D_{{22}} & 0 \\\\
        0 & 0 & D_{{33}}
        \\end{{bmatrix}}
        \\begin{{bmatrix}}
        \\epsilon_{{xx}} \\\\ \\epsilon_{{yy}} \\\\ \\gamma_{{xy}}
        \\end{{bmatrix}}
        $$"

    3. Coverage:
    - For each question, include a "coverage" field.
    - In this field, estimate the percentage of the possible answer that is covered by the input text.
    - Use your judgment to assign a realistic percentage in integer form, considering the depth and specificity of the input text.

    Generate as many questions as needed to cover the input text, up to {k} diverse questions, with Coverage 30-100 percentage.
    """

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Input text:\n{text}"}
        ],
        response_format=QResponse,
        temperature=0.15,
        max_tokens=512*k,               # Adjust as needed. Remove it to maximize the output length. (?)
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    
    response = completion.choices[0].message.parsed
        
    try:
        parsed_content = json.loads(response.json())['all_questions']
        return parsed_content
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        print("Generated content was:")
        print(response)
        return None

    

def gen_questions_s(client: OpenAI, text: str, k: int, model: str = "gpt-4o-mini") -> QResponse:
    system_prompt = f"""
    You are an AI assistant specialized in creating educational content for Finite Element Method (FEM).
    Generate comprehensive set of questions on topics related to FEM from the input text. **Only questions, no answer is needed.** Follow these guidelines:

    1. Questions:
    - Focus on fundamental concepts, theories, and general applications of FEM.
    - Ensure that the questions are relevant to the input text, and can be at least partially answered using the provided text.
    - Emphasize broad understanding rather than niche knowledge.
    - Questions can be of any length needed to fully express the concept being tested.
    - Complex questions involving multiple parts or mathematical derivations are encouraged.
    - Each question should have all the information needed such that it makes sense without referencing the input text. 
    - Any variables that are used in the question must be defined in the question.
    - Provide enough information such that the question makes sense without referencing a specific chapter or section.
    - Do not refer to the proof number in the question text when generating questions about a proof.
    - Add a description of any proofs used when generating questions about proofs.

    2. Coverage:
    - For each question, include a "coverage" field.
    - In this field, estimate the percentage of the possible answer that is covered by the input text.
    - Use your judgment to assign a realistic percentage in integer form, considering the depth and specificity of the input text.

    Note: Mathematical Notation:
    - Use LaTeX formatting for mathematical expressions
    - For inline equations, use single $ wrapper (e.g., "Calculate the strain energy $U = \\frac{1}{2}\\int_V \\sigma\\epsilon dV$")
    - For display equations, use double $$ wrapper, e.g.:
        "Derive the stiffness matrix given the following stress-strain relationship:
        $$
        \\begin{{bmatrix}}
        \\sigma_{{xx}} \\\\ \\sigma_{{yy}} \\\\ \\tau_{{xy}}
        \\end{{bmatrix}} =
        \\begin{{bmatrix}}
        D_{{11}} & D_{{12}} & 0 \\\\
        D_{{21}} & D_{{22}} & 0 \\\\
        0 & 0 & D_{{33}}
        \\end{{bmatrix}}
        \\begin{{bmatrix}}
        \\epsilon_{{xx}} \\\\ \\epsilon_{{yy}} \\\\ \\gamma_{{xy}}
        \\end{{bmatrix}}
        $$"

    Note: Your response format as JSON must adhere to the following structure:
    [
    {{
        "question": "What are the shape functions and their role in accuracy of approximations?",
        "coverage": 95
    }},
    {{
        "question": "How are boundary conditions imposed? Explain elimination approach.",
        "coverage": 70
    }}
    ]
    Generate as many questions as needed to cover the input text, up to {k} diverse questions, with Coverage 30-100 percentage.
    """

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Input text:\n{text}"}
        ],
        # response_format=QResponse,
        temperature=0.15,
        # max_tokens=512*k,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    
    response =  completion.choices[0].message.content
 
    # fix backslashes before parsing as JSON
    fixed_response = response.replace('\\', '\\\\')

    try:
        return json.loads(fixed_response)
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        print("Generated content was:")
        print(response)
        return []
        

def gen_answer(client: OpenAI, question: str, context: str, model: str = "gpt-4o-mini"):
    system_prompt = f"""
    You are an AI teaching assistant for a Finite Element Method (FEM) course. Answer questions based EXCLUSIVELY on the provided context. If context is insufficient for a very accurate answer, respond with: Answer: "NOT ENOUGH INFO."

    If context is sufficient:

    1. Answer Guidelines:
    - Use only information from the context
    - Restrict your use of finite element method knowledge to what is provided in the context provided. Do not use additional background finite element method knowledge in generating the answer (you may use background knowledge from other areas). 
    - Show step-by-step work for calculations
    - For multiple valid interpretations, provide separate answers

    2. Mathematical Notation:
    - Use $ for inline equations (e.g.,  $U = \\frac{{1}}{{2}} \\int_V \\sigma \\epsilon dV$)
    - Use $$ for display equations, especially matrices:
    $$
    \begin{{bmatrix}}
    \sigma_{{xx}} & \sigma_{{xy}} \\
    \sigma_{{yx}} & \sigma_{{yy}}
    \end{{bmatrix}}
    $$

    Note: Focus on FEM fundamentals, theories, and applications as presented in the context.
    """

    user_prompt = f"""
    Context:
    {context}

    Question:
    {question}

    Answer (based EXCLUSIVELY on the above context):
    """

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.15,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    response = completion.choices[0].message.content
    return  response