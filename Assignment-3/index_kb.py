# index_kb.py
"""
Index KB JSON into Pinecone using Azure OpenAI embeddings via LangChain.
"""

import os, json, argparse
from tqdm import tqdm
import pinecone
from langchain_openai import AzureOpenAIEmbeddings

def load_kb(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    entries = []
    for i, item in enumerate(data):
        entry_id = item.get("id") or f"KB{(i+1):03d}"
        text = item.get("text") or item.get("content") or item.get("snippet") or item.get("description", "")
        title = item.get("title", "")
        entries.append({"id": entry_id, "title": title, "text": text})
    return entries

def init_pinecone(index_name, api_key, environment, dimension):
    pinecone.init(api_key=api_key, environment=environment)
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=dimension, metric="cosine")
    return pinecone.Index(index_name)

def main(args):
    kb = load_kb(args.input)
    if not kb:
        raise SystemExit("KB is empty")

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.environ["AZURE_EMBED_DEPLOYMENT"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
    )

    vectors = []
    for item in tqdm(kb, desc="Embedding"):
        emb = embeddings.embed_query(f"{item['title']}\n\n{item['text']}")
        vectors.append((item["id"], emb, {"title": item["title"], "text": item["text"]}))

    index = init_pinecone(
        os.environ["PINECONE_INDEX"],
        os.environ["PINECONE_API_KEY"],
        os.environ["PINECONE_ENV"],
        len(vectors[0][1]),
    )

    # batch upserts
    for i in range(0, len(vectors), 50):
        index.upsert(vectors=vectors[i : i + 50])

    with open("index_report.json", "w", encoding="utf-8") as f:
        json.dump({"num_upserted": len(vectors)}, f, indent=2)

    print(f"✅ Indexed {len(vectors)} entries into Pinecone.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    args = p.parse_args()
    main(args)
