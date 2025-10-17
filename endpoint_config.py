from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import logging

from chunkcache import HybridCacheManager
from followupTool import QueryStateManager
from weaviate_ops import WeaviateOPS
from rag_minmax_ch import RAG_main

# ---------------------------------------------------
# Setup FastAPI + Logging
# ---------------------------------------------------
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global singletons (initialized at startup)
CACHE: HybridCacheManager = None
STATE: QueryStateManager = None
VSTORE: WeaviateOPS = None
RAG: RAG_main = None


# ---------------------------------------------------
# Lifecycle Events
# ---------------------------------------------------
@app.on_event("startup")
async def startup_event():
    global CACHE, STATE, VSTORE, RAG

    logger.info("🚀 Initializing global resources...")

    # Initialize cache and state managers
    CACHE = HybridCacheManager()
    STATE = QueryStateManager()

    # Initialize Weaviate connection (one-time)
    VSTORE = WeaviateOPS()

    # Initialize RAG model (one-time)
    RAG = RAG_main(
        doc_path=r"C:\Users\sruti\RAG_model\XYZ_doc.txt",
        collection_name="Collection5",
        model="sentence-transformers/paraphrase-MiniLM-L6-v2"
    )

    logger.info("✅ All resources initialized successfully.")


@app.on_event("shutdown")
async def shutdown_event():
    global VSTORE
    logger.info("🛑 Shutting down, closing Weaviate client...")
    if VSTORE:
        VSTORE.close()
        logger.info("✅ Weaviate client closed.")


# ---------------------------------------------------
# API Endpoints
# ---------------------------------------------------
@app.post("/_add_docs")
async def add_docs(payload: dict):
    texts = payload.get("texts", [])
    coll = payload.get("collection", "default_collection")

    if not texts:
        raise HTTPException(status_code=400, detail="texts required")

    # Create collection if needed and insert into Weaviate
    VSTORE.create_collection(coll)
    VSTORE.insert_chunks(texts)

    # Add chunks to in-memory cache too
    chunks = {f"doc_{i}": t for i, t in enumerate(texts)}
    CACHE.add_chunks(chunks)

    return {"inserted": len(texts), "collection": coll}


@app.get("/_conv/{conv_id}")
async def get_conv(conv_id: str):
    return STATE.get_answered(conv_id)


@app.post("/ask")
async def ask_question(payload: dict):
    query = payload.get("query", "")
    conv_id = payload.get("conv_id", "default")

    if not query:
        raise HTTPException(status_code=400, detail="query required")

    # Step 1: Check cache first
    cached = CACHE.retrieve_chunks(query, top_k=1)
    if cached:
        return {"answer": cached[0], "cached": True}

    # Step 2: Use RAG pipeline
    if RAG:
        resp = RAG.gen_response(query=query)
        return {"answer": resp, "cached": False}
    else:
        raise HTTPException(status_code=500, detail="RAG pipeline not initialized")

