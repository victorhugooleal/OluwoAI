import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import List, Dict

class OduLocalIndexer:
    def __init__(self, persist_directory: str = "data/vector_db"):
        self.persist_dir = persist_directory
    # ...implementação completa conforme instructions...
