"""
RAG system for intelligent resume chunking and retrieval.
Instead of passing the entire resume to LLM, we chunk it and retrieve only relevant sections.
"""
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import uuid
import re

from app.logger import setup_logger

logger = setup_logger(__name__)

# Disable TensorFlow before importing transformers (to avoid Keras 3 issue)
import os
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Try to import sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError as e:
    EMBEDDINGS_AVAILABLE = False
    logger.warning(f"sentence-transformers not available: {e}. Install with: pip install sentence-transformers")


class ResumeRAG:
    """
    RAG system for chunking resumes and retrieving relevant sections.
    This reduces token usage and improves processing speed.
    """

    def __init__(self, persist_directory: str = "./vectorstore"):
        """
        Initialize the Resume RAG system.

        Args:
            persist_directory: Directory to persist the vector database
        """
        if not EMBEDDINGS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for RAG. "
                "Install it with: pip install sentence-transformers"
            )

        logger.info(f"Initializing Resume RAG system")

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Initialize embedding model (lightweight and fast)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Loaded embedding model: all-MiniLM-L6-v2 (384 dimensions)")

        # Create collection for resume chunks
        self.collection = self.client.get_or_create_collection(
            name="resume_chunks",
            metadata={"description": "Resume chunks for RAG retrieval"}
        )

        logger.info("Resume RAG system initialized")

    def chunk_resume(self, resume_text: str) -> List[Dict[str, str]]:
        """
        Intelligently chunk a resume into meaningful sections.

        Args:
            resume_text: Full resume text

        Returns:
            List of chunks with metadata
        """
        chunks = []

        # Common resume section headers
        section_patterns = {
            'skills': r'(?i)(skills?|technical skills|competencies)',
            'experience': r'(?i)(experience|work experience|employment|professional experience)',
            'projects': r'(?i)(projects?|portfolio)',
            'education': r'(?i)(education|academic background)',
            'certifications': r'(?i)(certifications?|certificates?)',
            'summary': r'(?i)(summary|objective|profile)',
            'achievements': r'(?i)(achievements?|accomplishments?|awards?)',
        }

        # Split resume into lines
        lines = resume_text.split('\n')
        current_section = 'other'
        current_chunk = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this line is a section header
            is_header = False
            for section_name, pattern in section_patterns.items():
                if re.search(pattern, line):
                    # Save previous chunk if it exists
                    if current_chunk:
                        chunk_text = '\n'.join(current_chunk)
                        if len(chunk_text.strip()) > 20:  # Minimum chunk size
                            chunks.append({
                                'text': chunk_text,
                                'section': current_section,
                                'char_count': len(chunk_text)
                            })

                    # Start new section
                    current_section = section_name
                    current_chunk = [line]
                    is_header = True
                    break

            if not is_header:
                current_chunk.append(line)

        # Add the last chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            if len(chunk_text.strip()) > 20:
                chunks.append({
                    'text': chunk_text,
                    'section': current_section,
                    'char_count': len(chunk_text)
                })

        logger.info(f"Chunked resume into {len(chunks)} sections")
        for chunk in chunks:
            logger.info(f"  - {chunk['section']}: {chunk['char_count']} chars")

        return chunks

    def store_resume(self, resume_text: str, session_id: str) -> List[str]:
        """
        Chunk and store a resume in the vector database.

        Args:
            resume_text: Full resume text
            session_id: Unique session ID for this resume

        Returns:
            List of chunk IDs
        """
        logger.info(f"Storing resume for session: {session_id}")

        # Chunk the resume
        chunks = self.chunk_resume(resume_text)

        # Generate embeddings and store
        chunk_ids = []
        for chunk in chunks:
            # Generate embedding
            embedding = self.embedding_model.encode(chunk['text']).tolist()

            # Create unique ID
            chunk_id = f"{session_id}_{uuid.uuid4()}"

            # Store in ChromaDB
            self.collection.add(
                documents=[chunk['text']],
                embeddings=[embedding],
                metadatas=[{
                    'session_id': session_id,
                    'section': chunk['section'],
                    'char_count': chunk['char_count']
                }],
                ids=[chunk_id]
            )

            chunk_ids.append(chunk_id)

        logger.info(f"Stored {len(chunk_ids)} chunks for session {session_id}")
        return chunk_ids

    def retrieve_relevant_chunks(
        self,
        session_id: str,
        query: str,
        n_results: int = 3,
        section_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant resume chunks for a query.

        Args:
            session_id: Session ID of the resume
            query: What information we're looking for
            n_results: Number of chunks to retrieve
            section_filter: Optional filter by section (e.g., 'skills', 'experience')

        Returns:
            List of relevant chunks with metadata
        """
        logger.info(f"Retrieving chunks for query: '{query}' (session: {session_id})")

        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()

        # Build where filter
        where_filter = {'session_id': session_id}
        if section_filter:
            where_filter['section'] = section_filter

        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter
        )

        # Format results
        retrieved_chunks = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                retrieved_chunks.append({
                    'text': doc,
                    'section': results['metadatas'][0][i]['section'],
                    'char_count': results['metadatas'][0][i]['char_count'],
                    'distance': results['distances'][0][i] if results['distances'] else None
                })

        logger.info(f"Retrieved {len(retrieved_chunks)} relevant chunks")
        for chunk in retrieved_chunks:
            logger.info(f"  - Section: {chunk['section']}, Distance: {chunk['distance']:.4f}")

        return retrieved_chunks

    def get_all_chunks_for_session(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a session (fallback to full resume).

        Args:
            session_id: Session ID

        Returns:
            List of all chunks
        """
        results = self.collection.get(
            where={'session_id': session_id}
        )

        chunks = []
        if results['documents']:
            for i, doc in enumerate(results['documents']):
                chunks.append({
                    'text': doc,
                    'section': results['metadatas'][i]['section'],
                    'char_count': results['metadatas'][i]['char_count']
                })

        return chunks

    def delete_session(self, session_id: str):
        """
        Delete all chunks for a session.

        Args:
            session_id: Session ID to delete
        """
        # Get all IDs for this session
        results = self.collection.get(
            where={'session_id': session_id}
        )

        if results['ids']:
            self.collection.delete(ids=results['ids'])
            logger.info(f"Deleted {len(results['ids'])} chunks for session {session_id}")

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return {
            'total_chunks': self.collection.count(),
            'unique_sessions': len(set(
                meta['session_id']
                for meta in self.collection.get()['metadatas']
            )) if self.collection.count() > 0 else 0
        }


# Example usage functions
def example_usage():
    """Example of how to use the Resume RAG system."""

    # Initialize RAG system
    rag = ResumeRAG()

    # Sample resume text
    resume_text = """
    John Doe
    Software Engineer

    SUMMARY
    Experienced software engineer with 5 years in full-stack development.

    SKILLS
    Python, JavaScript, React, Node.js, PostgreSQL, Docker, AWS, Git

    EXPERIENCE
    Senior Software Engineer - Tech Corp (2021-Present)
    - Led development of microservices architecture
    - Improved system performance by 40%

    Software Engineer - Startup Inc (2019-2021)
    - Built React-based dashboard
    - Implemented CI/CD pipelines

    PROJECTS
    E-commerce Platform - Built scalable platform using React and Node.js
    ML Pipeline - Created data pipeline for ML model training

    EDUCATION
    B.S. Computer Science - University (2015-2019)
    """

    # Store the resume
    session_id = "user_123"
    chunk_ids = rag.store_resume(resume_text, session_id)
    print(f"Stored {len(chunk_ids)} chunks")

    # Retrieve relevant chunks for different queries
    print("\n--- Query 1: Skills ---")
    skills_chunks = rag.retrieve_relevant_chunks(
        session_id=session_id,
        query="What technical skills and technologies does the candidate know?",
        n_results=2
    )
    for chunk in skills_chunks:
        print(f"Section: {chunk['section']}")
        print(chunk['text'][:200])
        print()

    print("\n--- Query 2: Experience ---")
    exp_chunks = rag.retrieve_relevant_chunks(
        session_id=session_id,
        query="What is the candidate's work experience and achievements?",
        n_results=2
    )
    for chunk in exp_chunks:
        print(f"Section: {chunk['section']}")
        print(chunk['text'][:200])
        print()

    # Clean up
    rag.delete_session(session_id)


if __name__ == "__main__":
    example_usage()
