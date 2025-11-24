import os
import json
# import faiss # cause unknown "Segmentation fault: 11" error on CPU
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeException
from transformers import AutoTokenizer, AutoModel
import torch
import math
from openai import OpenAI
import nltk
from nltk import sent_tokenize
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv
from config import GPT_EMBED_SMALL_DIMENSION
import pandas as pd

nltk.download('punkt_tab')  # Download the sentence tokenizer

load_dotenv()


def get_openai_api_key():
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set. Please set it to continue.")
    return api_key


def get_pinecone_api_key():
    api_key = os.getenv('PINECONE_API_KEY')
    if not api_key:
        raise EnvironmentError("PINECONE_API_KEY environment variable is not set. Please set it to continue.")
    return api_key


def get_SBERT_embedding(text):

    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/paraphrase-MiniLM-L6-v2')

    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def get_GPT_embedding(text, model="text-embedding-3-small"):
    client = OpenAI(api_key=get_openai_api_key())
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding


def get_embedding(text):
    # must return list
    return get_GPT_embedding(text)


# Use FAISS
def load_or_create_faiss_index(index_path, ids_path, dimension):
    if os.path.exists(index_path) and os.path.exists(ids_path):
        index = faiss.read_index(index_path)
        with open(ids_path, "r") as f:
            ids = json.load(f)
    else:
        index = faiss.IndexFlatL2(dimension)
        ids = []
    return index, ids


def update_faiss_index(perspectives, index_path="database/faiss_index.idx", ids_path="database/faiss_ids.json"):
    new_vectors = []
    new_ids = []

    for perspective in perspectives:
        embedding = get_embedding(perspective['content'])
        new_vectors.append(embedding)
        new_ids.append(str(perspective['_id']))

    new_vectors = np.array(new_vectors).astype('float32')

    # Load or create the FAISS index
    dimension = new_vectors.shape[1]
    index, ids = load_or_create_faiss_index(index_path, ids_path, dimension)

    # Add new vectors to the index
    index.add(new_vectors)
    ids.extend(new_ids)

    # Save the updated index and IDs
    faiss.write_index(index, index_path)
    with open(ids_path, "w") as f:
        json.dump(ids, f)

    return


# Use Pinecone

def init_pinecone_index(index_name, dimension = GPT_EMBED_SMALL_DIMENSION):
    pinecone = Pinecone(api_key=get_pinecone_api_key())
    existing_indexes = [index['name'] for index in pinecone.list_indexes()]

    print("existing_indexes: ", existing_indexes)
    if index_name not in existing_indexes:
        try:
            pinecone.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        except PineconeException as e:
            print(e)
    else:
        print(f"Index {index_name} already exists.")

    return pinecone.Index(index_name)


def update_pinecone_index(pinecone_index, perspectives):

    vectors = []

    for perspective in perspectives:
        embedding = get_embedding(perspective['perspective'])
        vectors.append({
            "id": str(perspective['_id']),
            "values": embedding,
            "metadata": {
                "user_id": str(perspective['user_id']),
                "user_name": perspective['user_name'],
                "is_published": perspective['is_published']
            }
        })

    # Upsert vectors to Pinecone
    pinecone_index.upsert(vectors=vectors, namespace="perspectives")

def wordCount(content):
    # print(content)
    count = len(content.strip().split(" "))
    return count


def split(title, content, count = 2):
    #to do: 
    index = content.find(title) + len(title)
    actualContent = content[index:]
    wordCount = len(actualContent.strip().split(" "))
    
    paragraphs = actualContent.split("\n\n")

    if (wordCount < 1600 or len(paragraphs) <=3):
        return [content]
    else:
        # Split the article into paragraphs based on newline characters
        # Split the paragraphs in half
        mid_index = len(paragraphs) // 2
        overlapCount =  min(mid_index, 2)
        part1 = "\n\n".join(paragraphs[:mid_index+overlapCount])
        part1 = '[title:] ' + title + part1
        part2 = "\n\n".join(paragraphs[mid_index-overlapCount:])
        part2 = '[title:] ' + title + '[content:] ' +part2

        return  [part1, part2]
    
def splitBySentence(title, content, count = 2):
    #to do: 
    index = content.find(title) + len(title)
    actualContent = content[index:]
    wordCount = len(actualContent.strip().split(" "))
    
    sentences = sent_tokenize(actualContent)

    if (wordCount < 1600 or len(sentences) <=6):
        return [content]
    else:
        # Split the article into paragraphs based on newline characters
        # Split the paragraphs in half
        mid_index = len(sentences) // 2
        overlapCount =  min(mid_index, 3)
        part1 = "\n\n".join(sentences[:mid_index+overlapCount])
        part1 = '[title:] ' + title + part1
        part2 = "\n\n".join(sentences[mid_index-overlapCount:])
        part2 = '[title:] ' + title + '[content:] ' +part2

        return  [part1, part2]    
    
def splitContextIn3(content, count = 3):
    #to do: 
    actualContent = content
    wordCount = len(actualContent.strip().split(" "))
    
    sentences = sent_tokenize(actualContent)

    # Split the article into paragraphs based on newline characters
    # Split the paragraphs in half
    segCount = len(sentences) // count
    overlapCount =  min(segCount, 3)
    qList = []
    for index in range(count):
        if index == 0:
            qList.append("\n\n".join(sentences[:segCount+overlapCount]))
        elif index == count -1:
            qList.append( "\n\n".join(sentences[segCount* index -overlapCount:]))
        else: 
            qList.append("\n\n".join(sentences[segCount * index -overlapCount: segCount * (index + 1) + overlapCount]))

    return  qList
    
def scrapHFandCDC(URL):
    r = requests.get(URL)
    soup = BeautifulSoup(r.text, 'html.parser')

    #extract the main portion of the webpage
    #results = soup.find_all('main', attrs={'class':'container cdc-main'})
    results = soup.find_all('main', attrs={'class':'flex flex-1 flex-col'})


    #extract the content
    textList = []
    for hit in results:
    #hit = hit.text.strip()
        hit = hit.find_all(['h1', 'h2', 'h3', 'li', 'p'])
        text = [result.text.replace("\n", " ").replace("\t", "") for result in hit]
        textCombined = (" ".join(text) )
        textList.append(textCombined)


    if len(textList) > 1:
        resultText = (" ".join(textList) )
    else:
        resultText = textList[0]

    return resultText   

    