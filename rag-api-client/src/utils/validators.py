def validate_query(query: str) -> bool:
    if not isinstance(query, str):
        return False
    if len(query) == 0 or len(query) > 200:
        return False
    return True

def validate_document(document: dict) -> bool:
    if not isinstance(document, dict):
        return False
    if 'content' not in document or 'metadata' not in document:
        return False
    if not isinstance(document['content'], str) or not isinstance(document['metadata'], dict):
        return False
    return True

def validate_batch_documents(documents: list) -> bool:
    if not isinstance(documents, list):
        return False
    if len(documents) == 0 or len(documents) > 5:
        return False
    return all(validate_document(doc) for doc in documents)