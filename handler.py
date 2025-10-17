from typing import Dict, Any
from chunkcache import HybridCacheManager

async def tool_add_chunks(args: Dict[str, Any], cache: HybridCacheManager) -> Dict[str, Any]:
    chunks = args.get('chunks', {})
    cluster_id = args.get('cluster_id')
    cache.add_chunks(list(chunks.values()))
    return {"added": list(chunks.keys()), "cluster_id": cluster_id}

async def tool_retrieve_chunks(args: Dict[str, Any], cache: HybridCacheManager) -> Dict[str, Any]:
    query = args.get('query', '')
    top_k = int(args.get('top_k', 5))
    res = cache.retrieve_chunks(query, top_k=top_k)
    return {"chunks": [{"text": t} for t in res]}

TOOL_REGISTRY = {
    "add_chunks": {
        "handler": tool_add_chunks,
        "required": ["chunks"]
    },
    "retrieve_chunks": {
        "handler": tool_retrieve_chunks,
        "required": ["query"]
    }
}