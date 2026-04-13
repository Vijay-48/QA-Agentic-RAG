class RAGPipelineError(Exception):
    """Base exception for all RAG pipeline errors."""

class IngestionError(RAGPipelineError):
    """Raised when document loading or file reading fails."""

class ChunkingError(RAGPipelineError):
    """Raised when text chunking encounters an error."""

class EmbeddingError(RAGPipelineError):
    """Raised when embedding generation fails."""

class RetrievalError(RAGPipelineError):
    """Raised when vector store querying fails."""

class GenerationError(RAGPipelineError):
    """Raised when LLM response generation fails."""

class PipelineError(RAGPipelineError):
    """Raised when the overall RAG pipeline fails."""