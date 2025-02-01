from openai import OpenAI
import os
import csv
import re

client = OpenAI(api_key="ENTER YOUR API KEY HERE")

def read_file_content(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def generate_qa_from_openai(main_code, fem_code, assignment_description):
    prompt1 = r"""
    You are an expert in finite element methods (FEM), the deal.II library, and C++. You are tasked with creating detailed question-answer pairs for a coding assignment. The assignment description, along with the solution files (`main.cc` and `fem.h`), is provided. Follow these detailed instructions to generate the Q&A pairs:

    1. **Functions as Answers:** Each answer must include the implementation of individual functions or classes from the code files.

    2. **Cover All Code Components:** Generate questions for every function, constructor, destructor, and class definition in both `main.cc` and `fem.h`. Ensure that no code component is left out.

    3. **Detailed Question Context:** Each question must:
       - Include a **general problem statement** derived from the assignment description to provide a clear context.
       - Stand alone, without referencing the assignment, other questions, or answers, so that it makes sense independently.
       - Clearly ask for the specific function, constructor, destructor, or class related to the problem context.
       - Mention that the answer can use the open source library dealii

    4. **Variety in Questions:** In addition to asking for individual functions:
       - Include questions that require the entire class implementation as an answer (e.g., the `FEM` class).
       - Include a question asking for the names of all functions required to solve the assignment.

    5. **Formatting:** Use the following format for the Q&A pairs. Make sure not to number them:
       ```
       Q: <Insert detailed question here>
       A: <Insert complete function/class implementation here>
       ```
    6. **Descriptive Questions:** The questions should be long enough and verbose so that they are standalone and cover all the descriptive background from the original assignment without refering to the assignment.
    
    7. **Example Question for Context:** Use the style below as a reference for detailing each question:
       - Example Q:
         Consider the following differential equation of elastostatics, in strong form: \\ \\
         Find $u$ satisfying
         \begin{displaymath}
         (E\,A\, u_{,x})_{,x} + f\,A = 0,\quad \mbox{in}\; (0,L),
         \end{displaymath}
         \noindent for the following sets of boundary conditions and forcing
         function ($\bar{f}$ and $\hat{f}$ are constants):
         \begin{itemize}
          \setlength{\itemsep}{0pt}
          \item[(\romannumeral 1)]$u(0) = g_1$, $u(L) = g_2$, $f = \bar{f} x$,
          \item[(\romannumeral 2)]$u(0) = g_1$, $EAu_{,x} = h$ at $x = L$, $f = \bar{f} x$,
          \end{itemize}

         When writing a one-dimensional finite element code in C++ using the deal.II FEM library framework to solve the given problem, what will the class constructor look like?

       - Example A:
         Here is the class constructor to solve this problem:
         ```
         template <int dim>
         FEM<dim>::FEM (unsigned int order,unsigned int problem)
         : fe(FE_Q<dim>(QIterated<1>(QTrapez<1>(),order)), dim),
           dof_handler (triangulation)
         {
           basisFunctionOrder = order;
           prob = problem;
           for (unsigned int i=0; i<dim; ++i){
             nodal_solution_names.push_back("u");
             nodal_data_component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
           }
         }
         ```
    """

    prompt2 = f"""
    Here are the files related to the coding assignment:

    1. Assignment Description:
    {assignment_description}

    2. Contents of main.cc:
    {main_code}

    3. Contents of fem.h:
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
        {"tex": "codingAssignments/Assignment1/codingAssign1.tex", "main": "codingAssignments/Assignment1/main1.cc", "fem": "codingAssignments/Assignment1/FEM1.h"},
        {"tex": "codingAssignments/Assignment2/main.tex", "main": "codingAssignments/Assignment2/main2a.cc", "fem": "codingAssignments/Assignment2/FEM2a.h"},
        {"tex": "codingAssignments/Assignment2/main.tex", "main": "codingAssignments/Assignment2/main2a.cc", "fem": "codingAssignments/Assignment2/FEM2a_real.h"},
        {"tex": "codingAssignments/Assignment2/main.tex", "main": "codingAssignments/Assignment2/main2b.cc", "fem": "codingAssignments/Assignment2/FEM2b.h"},
        {"tex": "codingAssignments/Assignment3/CA3.tex", "main": "codingAssignments/Assignment3/main3.cc", "fem": "codingAssignments/Assignment3/FEM3.h"},
        {"tex": "codingAssignments/Assignment4/CA4.tex", "main": "codingAssignments/Assignment4/main4.cc", "fem": "codingAssignments/Assignment4/FEM4.h"},
        {"tex": "codingAssignments/Assignment4/CA4.tex", "main": "codingAssignments/Assignment4/main4.cc", "fem": "codingAssignments/Assignment4/L2norm4.py"},
        {"tex": "codingAssignments/Assignment5/codingAssign5.tex", "main": "codingAssignments/Assignment5/main5.cc", "fem": "codingAssignments/Assignment5/FEM5.h"},
    ]

    qa_output_csv = "codingQAPairs1-4o.csv"
    raw_output_txt = "codingQAPairsContent1-4o.txt"

    open(raw_output_txt, "w").close() # Clear existing content from the output files
    open(qa_output_csv, "w").close()

    for group in assignments_with_files:
        assignment_description = read_file_content(group["tex"])
        main_code = read_file_content(group["main"])
        fem_code = read_file_content(group["fem"])
        qa_pairs = generate_qa_from_openai(main_code, fem_code, assignment_description)

        save_to_csv(qa_pairs, qa_output_csv)
        print(f"Q&A pairs saved to {qa_output_csv}")
        save_to_txt(qa_pairs, raw_output_txt)
        print(f"Raw content saved to {raw_output_txt}")

if __name__ == "__main__":
    main()
