from weaviate_ops import WeaviateOPS
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from collections import deque
import numpy as np
import re
import logging
import os

from chunkcache import HybridCacheManager

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['API_KEY']= '<enter your api key>'
def query_gen():
    q = input("Enter query: ")
    return q

def clean(text: str) -> list[str]:
    text = re.sub(r"\n+", " ", text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    logging.info(f"sentences : {sentences}")
    return [s.strip() for s in sentences if s.strip()]

class RAG_main:
    def __init__(self,doc_path,collection_name:str, model="sentence-transformers/paraphrase-MiniLM-L6-v2"):
        self.doc_path=doc_path
        self.model=model
        self.chunks=[]
        self.encoder=SentenceTransformer(self.model)
        self.vectorstore=WeaviateOPS()
        self.collection_name=collection_name
        self.history=deque(maxlen=100)

        self.cache = HybridCacheManager(model=self.model)

    def chunking(self, sentences: list[str], fixed_threshold: float = 0.8, c: float = 0.9,
                 init_constant: float = 1.5) -> list[list[str]]:
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        embeddings = self.encoder.encode(sentences)
        chunks = []
        current_chunk = [sentences[0]]
        cluster_start, cluster_end = 0, 1
        pairwise_min = -float("inf")
        for i in range(1, len(sentences)):
            cluster_embeddings = embeddings[cluster_start:cluster_end]
            if cluster_end - cluster_start > 1:
                new_similarities = cosine_similarity(embeddings[i].reshape(1, -1), cluster_embeddings)[0]
                adjusted_threshold = pairwise_min * c * sigmoid((cluster_end - cluster_start) - 1)
                new_similarity = np.max(new_similarities)
                pairwise_min = min(np.min(new_similarities), pairwise_min)
            else:
                adjusted_threshold = 0
                pairwise_min = cosine_similarity(embeddings[i].reshape(1, -1), cluster_embeddings)[0][0]
                new_similarity = init_constant * pairwise_min

            if new_similarity > max(adjusted_threshold, fixed_threshold):
                current_chunk.append(sentences[i])
                cluster_end += 1
            else:
                chunks.append(current_chunk)
                current_chunk = [sentences[i]]
                cluster_start, cluster_end = i, i + 1
                pairwise_min = -float("inf")
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def load_doc(self):
        logging.info(f"Loading document- {self.doc_path}")
        if self.doc_path.endswith(".txt"):
            loader = TextLoader(self.doc_path, encoding="utf-8")
        else:
            loader = PyPDFLoader(self.doc_path)
        documents = loader.load()
        all_text = " ".join([doc.page_content for doc in documents])
        sentences = clean(all_text)
        logging.info(f"Splitting into {len(sentences)} sentences")
        self.chunks = self.chunking(sentences)
        logging.info(f"Chunk count: {len(self.chunks)} ")

    def store(self):
        logging.info(f"Storing document- {self.doc_path}")
        try:
            if self.collection_name is not None:
                self.vectorstore.create_collection(self.collection_name)

            texts=[" ".join(chunk) for chunk in self.chunks]
            self.vectorstore.insert_chunks(texts)
            self.cache.add_chunks(texts)
            logging.info("chunks stored in weaviate DB")
        except Exception as e:
            logging.error(e)

    def retrieve(self,query:str,top_k:int=7):
        logging.info(f"Retrieving document- {self.doc_path}")
        try:
            cached = self.cache.retrieve_chunks(query, top_k)
            if cached:
                logging.info("Cache hit, returning cached chunks.")
                return cached
            if not self.vectorstore:
                raise ValueError("Vectorstore not initialized. Store() must be called.")
            results=self.vectorstore.query_similar_chunks(query=query,limit=top_k)
            chunks = [obj.properties.get("title", "") for obj in results.objects]
            for i,c in enumerate(chunks,1):
                logging.info(f"\t{i}.\t{c}")
            return chunks
        except Exception as e:
            logging.error(f"Error retrieving chunks- {e}")
            return []

    def gen_response(self,query:str,top_k:int=7)->str:
        logging.info(f"Generating response using llm")
        docs=self.retrieve(query,top_k)
        #retrieved_chunks = [doc.page_content for doc in docs if doc and hasattr(doc, "page_content")]
        context="\n\n".join(docs)
        history_deque="\n\n".join([f"User:{i['query']}\nAssistant:{i['response']}" for i in self.history])
        llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash",api_key=API_KEY)
        #support = "For the most current information on eligibility requirements, refer to the B9 Terms of Service or contact B9 customer support.See Terms of Services or contact B9 via support@bnine.com for additional terms, conditions, and eligibility requirements."

        prompt = f"""
        You are a helpful and professional banking assistant.
Always answer ONLY using the provided context. 
If the answer is not in the context, say you don’t have enough information. 
Use the previous conversation to maintain continuity and avoid repeating definitions unnecessarily. 

If the user’s query is broad or missing details, ask clarifying questions before answering. 
If the query can have multiple aspects (like eligibility, documents required, loan tenure), 
guide the user by suggesting what additional information they may want to know. 

Previous conversation:
{history_deque}

Retrieved context (from documents):
{context}

User's question:
{query}

Instructions:
- Answer concisely and clearly.
- If related details (like eligibility, conditions, documents) are found in context, include them.
- If details are missing, ask the user a follow-up question to narrow down the query.


        """

        response=llm.invoke(prompt)
        answer=response.content.strip()
        self.history.append({"query":query,"response":answer})
        logging.info("Response generated.")
        return answer

if __name__ == "__main__":
    doc_path = r"C:\Users\sruti\RAG_model\XYZ_doc.txt"
    coll_name = "Collection5"
    load_dotenv()
    API_KEY = os.getenv("API_KEY")
    rag = RAG_main(doc_path,coll_name, model="sentence-transformers/paraphrase-MiniLM-L6-v2")
    rag.load_doc()
    rag.store()
    while True:
        query = input("\nEnter your query (or type 'exit' to quit): ")
        if query.lower() in ["exit", "quit"]:
            print("Exiting...")
            break
        resp = rag.gen_response(query=query, top_k=7)
        print("\nGenerated Response:\n", resp)
    rag.vectorstore.close()

