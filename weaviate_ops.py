from sentence_transformers import SentenceTransformer
import weaviate
import weaviate.classes.config as wvc
from weaviate.classes.query import MetadataQuery
import logging

from typing import List, Dict, Union
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeaviateOPS:
    def __init__(self, mode: str = "transformers"):
        self.mode = mode
        self.client = weaviate.connect_to_local()
        self.collection = None
        self.collection_title=None

        if not self.client.is_ready():
            raise RuntimeError("Weaviate client is not ready.")

        logger.info("Weaviate client connected successfully.")

    def create_collection(self, collection_title: str) -> None:
        self.collection_title = collection_title
        try:
            vectorizer_cfg = (
                wvc.Configure.Vectorizer.text2vec_transformers()
                if self.mode == "transformers" else None
            )

            self.collection = self.client.collections.create(
                name=collection_title,
                vectorizer_config=vectorizer_cfg,
                properties=[
                    wvc.Property(name="title", data_type=wvc.DataType.TEXT),
                    wvc.Property(name="meta_mode", data_type=wvc.DataType.TEXT),
                    wvc.Property(name="doc_type", data_type=wvc.DataType.INT),
                ]
            )
            logger.info(f"Collection '{collection_title}' created successfully.")

        except Exception as e:
            if "already exists" in str(e).lower():
                logger.warning(f"Collection '{self.collection_title}' already exists. Reusing it.")
                self.collection = self.client.collections.get(self.collection_title)
            else:
                raise RuntimeError(f"Failed to create collection: {str(e)}")

    def insert_chunks(self, chunks: List[Union[str, Dict]]) -> None:
        if self.collection is None:
            self.collection = self.client.collections.get(self.collection_title)

        inserted = 0
        for chunk in chunks:
            if isinstance(chunk, dict):
                text = chunk.get("text") or chunk.get("title")
                metadata = {k: v for k, v in chunk.items() if k not in ("text", "title")}
                metadata["title"] = text
            else:
                text = chunk
                metadata = {"title": text}
            vector=None
            self.collection.data.insert(
                properties=metadata,
                vector=vector if vector else None
            )
            inserted += 1

        logger.info(f"Inserted {inserted} chunks into collection '{self.collection_title}'.")

    def query_similar_chunks(self, query: str, limit: int = 7):
        if self.collection is None:
            self.collection = self.client.collections.get(self.collection_title)
        response = self.collection.query.near_text(
            query=query,
            limit=limit,
            return_metadata=MetadataQuery(distance=True) # type: ignore
        )

        logger.info(f"Retrieved {len(response.objects)} results for query '{query}'.")

        for obj in response.objects:
            logger.info(f"Result: {obj.properties}, Similarity Score: {obj.metadata.distance}")

        return response

    def delete_collection(self) -> None:
        try:
            self.client.collections.delete(self.collection_title)
            logger.info(f"Collection '{self.collection_title}' deleted successfully.")
        except Exception as e:
            logger.error(f"Failed to delete collection: {str(e)}")


    def delete_by_metadata(self, doc_type:str):
        filter_by = self.collection.query.fetch_objects(
            filters={
                "path": ["doc_type"],
                "operator": "Equal",
                "valueInt": doc_type
            },
            return_metadata=False
        )
        for obj in filter_by.objects:
            self.collection.data.delete_by_id(obj.uuid)
        logger.info(f"Chunks under '{self.collection_title}' deleted successfully.")

    def get_metadata(self):
        collection=self.client.collections.get(self.collection_title)
        objects=collection.query.fetch_objects(limit=10000)
        doc_types=set()
        total_chunks=0
        sentence_count=0
        for o in objects.objects:
            properties=o.properties
            text=properties.get("text","") or properties.get("title","")
            doc_type=properties.get("doc_type","unknown")
            doc_types.add(doc_type)
            total_chunks += 1
            sentence_count+=len([s for s in text.split('.') if s.strip()])
        return {
            "doc_type":list(doc_types),
            "total_chunks":total_chunks,
            "sentence_count":sentence_count,
        }

    def close(self) -> None:
        self.client.close()
        logger.info("Weaviate client connection closed.")



