from openai import OpenAI
import os
import csv
import re

client = OpenAI(api_key="ENTER YOUR API KEY HERE")

def read_file_content(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def generate_qa_from_openai(femCode, assignmentDescription):
    prompt1 = r"""
    You are an expert in finite element methods (FEM), the fenics library, and python. You are tasked with creating detailed question-answer pairs for a coding assignment. The assignment description, along with the solution file (`fem.h`), is provided. Follow these detailed instructions to generate the Q&A pairs:

    1. **Answers Based On Code:** Answers should be based on code implementation.

    2. **Cover All Code Components:** Generate as many questions using `fem.h` ensuring no code component is left out. More the questions, the better. It is ok if some questions are repeated/have some overlap.

    3. **Detailed Question Context:** Each question must:
       - Include a **general problem statement** derived from the assignment description to provide a clear context.
       - Stand alone, without referencing the assignment, other questions, or answers, so that it makes sense independently.
       - Clearly ask for the specific code implementation related to the problem context.
       - Mention that the answer should be based on open source finite element library fenics
       - The questions should be long enough and verbose so that they are standalone and cover all the descriptive background from the original assignment without refering to the assignment.
       - If the assignment asks for something particular to be implemented such as the boundary condition (pde variables, mesh variables etc), the question should list the boundary conditions to be implemented.
       - Make sure to not refer to the assignment.

    4. **Formatting:** Use the following format for the Q&A pairs. Make sure not to number them:
       ```
       Q: <Insert detailed question here>
       A: <Insert complete function/class implementation here>
       ```
    """

    prompt2 = f"""
    Here are the files related to the coding assignment:

    1. Assignment Description:
    {assignmentDescription}

    2. Contents of fem.h:
    {femCode}
    """

    prompt = prompt1 + prompt2

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            #{"role": "system", "content": "You are an expert in finite element method, deal.II, and C++."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=16384 #max output tokens from OpenAI
    )
    return response.choices[0].message.content.strip()


def save_to_csv(qa_pairs, output_file):
    qa_list = re.split(r'(Q:|A:)', qa_pairs)
    
    qa_data = []
    current_question = None
    for i in range(len(qa_list)):
        if qa_list[i] == "Q:":
            current_question = qa_list[i + 1].strip()
        elif qa_list[i] == "A:" and current_question:
            current_answer = qa_list[i + 1].strip()
            qa_data.append({"Question": current_question, "Answer": current_answer})
            current_question = None

    file_exists = os.path.exists(output_file) and os.path.getsize(output_file) > 0

    with open(output_file, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["Question", "Answer"])
        if not file_exists:
            writer.writeheader()  # Write the header only if the file is new or empty
        writer.writerows(qa_data)

def save_to_txt(content, output_file):
    with open(output_file, mode="a", encoding="utf-8") as file:
        file.write(content + "\n\n")

def main():
    assignments_with_files = [
        {"tex": "fenicsCodingAssignments/Assignment1/codingAssign1.tex", "fem": "fenicsCodingAssignments/Assignment1/fem1.py"},
        {"tex": "fenicsCodingAssignments/Assignment2/codingAssign2.tex", "fem": "fenicsCodingAssignments/Assignment2/fem2a1.py"},
        {"tex": "fenicsCodingAssignments/Assignment2/codingAssign2.tex", "fem": "fenicsCodingAssignments/Assignment2/fem2a2.py"},
        {"tex": "fenicsCodingAssignments/Assignment2/codingAssign2.tex", "fem": "fenicsCodingAssignments/Assignment2/fem2c.py"},
        {"tex": "fenicsCodingAssignments/Assignment3/CA3.tex", "fem": "fenicsCodingAssignments/Assignment3/fem3.py"},
        {"tex": "fenicsCodingAssignments/Assignment4/CA4.tex", "fem": "fenicsCodingAssignments/Assignment4/fem4.py"},
        {"tex": "fenicsCodingAssignments/Assignment5/codingAssign5.tex", "fem": "fenicsCodingAssignments/Assignment5/fem5.py"},
    ]

    qa_output_csv = "fenicsCodingQAPairs5-4o.csv"
    raw_output_txt = "fenicsCodingQAPairsContent5-4o.txt"

    
    open(raw_output_txt, "w").close() # Clear existing content from the output files
    open(qa_output_csv, "w").close()

    for group in assignments_with_files:
        assignmentDescription = read_file_content(group["tex"])
        femCode = read_file_content(group["fem"])
        qa_pairs = generate_qa_from_openai(femCode, assignmentDescription)


        save_to_csv(qa_pairs, qa_output_csv)
        print(f"Q&A pairs saved to {qa_output_csv}")
        save_to_txt(qa_pairs, raw_output_txt)
        print(f"Raw content saved to {raw_output_txt}")

if __name__ == "__main__":
    main()
