import os
import io
from PIL import Image
import pytesseract
from wand.image import Image as wi
import gc
import pandas as pd
import re
import json
from openai import OpenAI
from llamaapi import LlamaAPI
#import nltk
#nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer

client = OpenAI(api_key="ENTER YOUR API TOKEN",)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", padding_side="left", cache_dir="/home1/rahulgul/llm1", return_tensors="pt")

def split_text_with_overlap(text, max_tokens=512, overlap_tokens=25):
    if not text:
        print('Error: The text to be tokenized is a None type.')
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_tokens = 0
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        sentence_token_count = len(sentence_tokens)
        if current_tokens + sentence_token_count > max_tokens:
            chunks.append(' '.join(current_chunk))
            overlap_chunk = []
            overlap_chunk_tokens = 0
            while current_chunk and overlap_chunk_tokens < overlap_tokens:
                last_sentence = current_chunk.pop()
                overlap_chunk.insert(0, last_sentence)
                overlap_chunk_tokens += len(tokenizer.encode(last_sentence, add_special_tokens=False))
            current_chunk = overlap_chunk
            current_tokens = overlap_chunk_tokens
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_token_count
            i += 1
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def Get_text_from_image(pdf_path):
    pdf=wi(filename=pdf_path,resolution=300)
    pdfImg=pdf.convert('jpeg')
    imgBlobs=[]
    extracted_text=[]
    for img in pdfImg.sequence:
        page=wi(image=img)
        imgBlobs.append(page.make_blob('jpeg'))
    for imgBlob in imgBlobs:
        im=Image.open(io.BytesIO(imgBlob))
        text=pytesseract.image_to_string(im,lang='eng')
        extracted_text.append(text)
    return (extracted_text)

def generate_question_answer_pairs(chunk):
    prompt = "You have a section from a book on the Finite Element Method (FEM) and want to generate as many curated question-and-answer pairs as necessary to fully cover the content of the section. Based on your knowledge of finite element analysis, correct the equations if incorrect and replace the symbols with the standard symbols used in literature. The questions should focus on explanations that assess understanding of the concepts, techniques, and principles presented in the material, rather than on referencing the book directly. Answers should be thorough, clear, and emphasize learning and knowledge retention. The goal is to create questions that encourage deep understanding and contextual learning of FEM, without relying on exact book references. Once the question-answer pairs are generated, double check that they do not refer to the book, equation number, section number etc. For each equation, make sure to write them in latex script as \begin{equation} … \end{equation}. For every greek and latin symbol, make sure to write them in latex script as $ $. Remove any unnecessary * or - at the beginning of end of the pairs. For clarity and ease of organization, format the question-answer pairs in the following way: \n Q: <Insert question here> \n A: <Insert detailed answer here> \n\n" + f"Section: {chunk}"
    response = client.chat.completions.create(   
        model="gpt-4o-mini",
        messages=[
            #{"role": "system", "content": "You are an expert in finite element method."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=2500
    )
    return response.choices[0].message.content.strip() 

def save_qna_to_excel(chunks, output_path):
    qna_pairs = []    
    qna_chunck_pairs = []
    for chunk in chunks:
        qna = generate_question_answer_pairs(chunk)
        qna_chunck_pairs.append(qna)
        qna_list = re.split(r'Q:|A:', qna)  
        for i in range(1, len(qna_list) - 1, 2):  
            question = qna_list[i].strip()
            answer = qna_list[i + 1].strip()
            qna_pairs.append((question, answer))
    df = pd.DataFrame(qna_pairs, columns=["Question", "Answer"])
    df.to_csv(output_path, index=False)    
    df_chunck = pd.DataFrame({'Chunks': chunks, 'Q&A Pairs': qna_chunck_pairs})
    df_chunck.to_csv('qna_chunck_pairs1.csv', index=False)

text_from_book = Get_text_from_image("../pdf2/Fem_book2/1.pdf")
raw_text = ' '.join(text_from_book)
#raw_text=re.sub('[^A-Za-z0-9.]+', ' ',raw_text)
print("text_length=",len(raw_text))
chunks = split_text_with_overlap(raw_text, max_tokens=512)
print("Creating Question-Answer pairs")
save_qna_to_excel(chunks, 'qna_pairs1.csv')
print("Job completed")
