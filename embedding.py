from langchain_huggingface import HuggingFaceEmbeddings
from concurrent.futures import ThreadPoolExecutor

def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1.5",
        model_kwargs={"trust_remote_code": True}
    )

