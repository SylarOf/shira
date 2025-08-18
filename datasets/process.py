import requests
from bs4 import BeautifulSoup
from embeddings.embeddings import embed_document,embed_query,cosine_similarity

# data should have 4 keys: question,answer,files,triplets
# files that the text to invoke for llm

def get_text_from_website(url:str)->str:
    headers = {"User-Agent":"Mozilla/5.0"} 
    resp = requests.get(url,headers=headers)
    if resp.status_code != 200:
        raise Exception(f"request failed,status code:{resp.status_code}")
    
    soup = BeautifulSoup(resp.text,"html.parser")
    return soup.get_text()

def get_text_from_dir(file_path:str,encoding='utf-8')->str:
    with open(file_path,'r',encoding=encoding) as f:
        content = f.read()
        return content


def split_text(text:str,chunk_size:int = 500,overlap:int = 50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

    

def get_topk_files(data,k):
    urls = data['websites']
    files = data['files']

    raw_data = []
    for url in urls:
        raw_data.append(get_text_from_website(url))
    
    for file in files:
        raw_data.append(get_text_from_dir(file))


    query_embedding = embed_query(data['question'])
    documents = []
    for split in raw_data:
        document_emebedding = embed_document(raw_data)
        score = cosine_similarity(document_emebedding,query_embedding)
        documents.append({"file":split, "score":score})


    documents.sort(key=lambda x: x[score],reverse=True)

    return [file for file, _ in documents[:k]]


    
        

