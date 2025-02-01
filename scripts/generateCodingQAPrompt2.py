from openai import OpenAI
import os
import csv
import re

client = OpenAI(api_key="ENTER YOUR API KEY HERE")

def read_file_content(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def generate_qa_from_openai(main_code, fem_code, assignment_description, templateMain, templateFEM):
    prompt1 = r"""
    You are an expert in finite element methods (FEM), the deal.II library, and C++. You are tasked with creating detailed question-answer pairs for a coding assignment. The assignment description and the coding template file, along with the solution files (`main.cc` and `fem.h`), are provided. Follow these detailed instructions to generate the Q&A pairs:

    1. **Test on identical material/information as the provided assignment template:** Question Answer pairs must be based on what the coding assignment is targeting the student to understand. The student is expected to use the template coding files and fill them to get the solution coding files. Match the differences between the coding template files and the coding solution and base your question-answers on this. Essentially the QA pairs generated should quiz the student on the identical material tested by the coding assignment and the provide coding template.

    2. **Detailed Question Context:** Each question must:
       - Include a **general problem statement** derived from the assignment description to provide a clear context.
       - Stand alone, without referencing the assignment, other questions, or answers, so that it makes sense independently.
       - Clearly ask for the specific function, constructor, destructor, or class related to the problem context as in the previous point.
       - Mention that the answer can use the open source library dealii
       - The questions should be long enough and verbose so that they are standalone and cover all the descriptive background from the original assignment without refering to the assignment.
       - If the assignment asks for something particular to be implemented such as the boundary condition (pde variables, mesh variables etc), the question should list the boundary conditions to be implemented.
    3. **Generate as many questions:** Cover all the assignment problem specific implementations in the code even if they are already provided in the template files.
    
    4. **Formatting:** Use the following format for the Q&A pairs. Make sure not to number them:
       ```
       Q: <Insert detailed question here>
       A: <Insert function/class implementation here>

    
    """

    prompt2 = f"""
    Here are the files related to the coding assignment:

    1. Assignment Description:
    {assignment_description}
    
    2. Contents of Template main.cc:
    {templateMain}
    
    3. Contents of template fem.h:
    {templateFEM}

    4. Contents of solution main.cc:
    {main_code}

    5. Contents of solution fem.h:
    {fem_code}
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
        {"tex": "codingAssignments/Assignment1/codingAssign1.tex", "main": "codingAssignments/Assignment1/main1.cc", "fem": "codingAssignments/Assignment1/FEM1.h", "templateMain": "codingAssignments/Assignment1/templatemain1.cc", "templateFEM": "codingAssignments/Assignment1/templateFEM1.h"},
        {"tex": "codingAssignments/Assignment2/main.tex", "main": "codingAssignments/Assignment2/main2a.cc", "fem": "codingAssignments/Assignment2/FEM2a.h", "templateMain": "codingAssignments/Assignment2/templatemain2a.cc", "templateFEM": "codingAssignments/Assignment2/templateFEM2a.h"},
        {"tex": "codingAssignments/Assignment2/main.tex", "main": "codingAssignments/Assignment2/main2a.cc", "fem": "codingAssignments/Assignment2/FEM2a_real.h", "templateMain": "codingAssignments/Assignment2/templatemain2a.cc", "templateFEM": "codingAssignments/Assignment2/templateFEM2a.h"},
        {"tex": "codingAssignments/Assignment2/main.tex", "main": "codingAssignments/Assignment2/main2b.cc", "fem": "codingAssignments/Assignment2/FEM2b.h", "templateMain": "codingAssignments/Assignment2/templatemain2b.cc", "templateFEM": "codingAssignments/Assignment2/templateFEM2b.h"},
        {"tex": "codingAssignments/Assignment3/CA3.tex", "main": "codingAssignments/Assignment3/main3.cc", "fem": "codingAssignments/Assignment3/FEM3.h", "templateMain": "codingAssignments/Assignment3/templatemain3.cc", "templateFEM": "codingAssignments/Assignment3/templateFEM3.h"},
        {"tex": "codingAssignments/Assignment4/CA4.tex", "main": "codingAssignments/Assignment4/main4.cc", "fem": "codingAssignments/Assignment4/FEM4.h", "templateMain": "codingAssignments/Assignment4/templatemain4.cc", "templateFEM": "codingAssignments/Assignment4/templateFEM4.h"},
        {"tex": "codingAssignments/Assignment4/CA4.tex", "main": "codingAssignments/Assignment4/main4.cc", "fem": "codingAssignments/Assignment4/L2norm4.py", "templateMain": "codingAssignments/Assignment4/templatemain4.cc", "templateFEM": "codingAssignments/Assignment4/templateFEM4.h"},
        {"tex": "codingAssignments/Assignment5/codingAssign5.tex", "main": "codingAssignments/Assignment5/main5.cc", "fem": "codingAssignments/Assignment5/FEM5.h", "templateMain": "codingAssignments/Assignment5/templatemain5.cc", "templateFEM": "codingAssignments/Assignment5/templateFEM5.h"},
    ]

    qa_output_csv = "codingQAPairsB3-4o.csv"
    raw_output_txt = "codingQAPairsContentB3-4o.txt"

    
    open(raw_output_txt, "w").close() # Clear existing content from the output files
    open(qa_output_csv, "w").close()

    for group in assignments_with_files:
        assignmentDescription = read_file_content(group["tex"])
        mainCode = read_file_content(group["main"])
        femCode = read_file_content(group["fem"])
        templateMainCode = read_file_content(group["templateMain"])
        templateFemCode = read_file_content(group["templateFEM"])
        qa_pairs = generate_qa_from_openai(mainCode, femCode, assignmentDescription, templateMainCode, templateFemCode)


        save_to_csv(qa_pairs, qa_output_csv)
        print(f"Q&A pairs saved to {qa_output_csv}")
        save_to_txt(qa_pairs, raw_output_txt)
        print(f"Raw content saved to {raw_output_txt}")

if __name__ == "__main__":
    main()
