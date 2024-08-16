def prepare_qa_data_for_llm(csv_file, coverage_threshold=0):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Filter rows based on coverage threshold
    filtered_df = df[df['coverage'] > coverage_threshold]
    
    # Prepare the data in the specified format for LLM fine-tuning
    qa_data = []
    for _, row in filtered_df.iterrows():
        qa_pair = {
            "input": f"[CONTEXT] {row['context']} \n[QUESTION] {row['question']}",
            "output": row['answer']
        }
        qa_data.append(qa_pair)
    
    return qa_data