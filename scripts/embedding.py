import json
import numpy as np

def get_embeddings(client, texts, model="text-embedding-3-small"):   
        
    # edge case -- `texts` must be a list, in case a single string is passed
    if isinstance(texts, str):
        texts = [texts]

    response = client.embeddings.create(
        input=texts,
        model=model
    )

    try:
        data = json.loads(response.json())['data']
        return [item['embedding'] for item in data]
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        print("Generated content was:")
        print(response)
        return [[]]
    
def fixed_knn_retrieval(question_embedding, context_embeddings, top_k=5):

    question_embedding = question_embedding / np.linalg.norm(question_embedding)
    context_embeddings = context_embeddings / np.linalg.norm(context_embeddings, axis=1, keepdims=True)

    similarities = np.dot(context_embeddings, question_embedding)
    sorted_indices = np.argsort(similarities)[::-1]
    selected_indices = sorted_indices[:top_k].tolist()
    return selected_indices