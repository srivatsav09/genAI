"""
Job RAG system for efficient job posting storage and semantic matching.
Chunks career pages by individual job postings and enables semantic search.
"""
from typing import List, Dict, Any, Optional
import uuid
import re

from app.logger import setup_logger

logger = setup_logger(__name__)

# Disable TensorFlow before importing transformers
import os
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Try to import dependencies
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("chromadb not installed. Install with: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError as e:
    EMBEDDINGS_AVAILABLE = False
    logger.warning(f"sentence-transformers not available: {e}")


class JobRAG:
    """
    RAG system for job postings with semantic search capabilities.

    Stores individual job postings as chunks in ChromaDB and enables
    semantic search to find jobs matching candidate skills.
    """

    def __init__(self, persist_directory: str = "./vectorstore_jobs"):
        """
        Initialize JobRAG system.

        Args:
            persist_directory: Directory to persist the vector database

        Raises:
            ImportError: If required dependencies are not installed
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb is required. Install with: pip install chromadb")

        if not EMBEDDINGS_AVAILABLE:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")

        logger.info("Initializing Job RAG system")

        # Initialize ChromaDB client
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Loaded embedding model: all-MiniLM-L6-v2 ({embedding_dim} dimensions)")

        # Create or get collection for jobs
        self.collection = self.client.get_or_create_collection(
            name="job_postings",
            metadata={"description": "Job postings with semantic search"}
        )

        logger.info("Job RAG system initialized")

    def store_jobs(self, jobs: List[Dict[str, Any]], session_id: str) -> int:
        """
        Store job postings in the vector database.

        Args:
            jobs: List of job dictionaries with keys like 'role', 'skills', 'description', etc.
            session_id: Unique session identifier for this batch of jobs

        Returns:
            Number of jobs stored
        """
        if not jobs:
            logger.warning("No jobs to store")
            return 0

        logger.info(f"Storing {len(jobs)} jobs with session_id: {session_id}")

        documents = []
        metadatas = []
        ids = []

        for idx, job in enumerate(jobs):
            # Create searchable text representation of job
            job_text = self._format_job_for_storage(job)
            documents.append(job_text)

            # Store metadata
            metadata = {
                "session_id": session_id,
                "job_index": idx,
                "role": job.get('role', 'Unknown'),
                "company": job.get('company', 'Unknown'),
            }

            # Add optional fields if present
            if 'experience' in job:
                metadata['experience'] = str(job['experience'])
            if 'location' in job:
                metadata['location'] = job.get('location', '')

            metadatas.append(metadata)

            # Generate unique ID
            job_id = f"{session_id}_job_{idx}"
            ids.append(job_id)

        # Generate embeddings
        embeddings = self.embedding_model.encode(documents).tolist()

        # Store in ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        logger.info(f"Successfully stored {len(jobs)} jobs")
        return len(jobs)

    def _format_job_for_storage(self, job: Dict[str, Any]) -> str:
        """
        Format job dictionary into searchable text.

        Args:
            job: Job dictionary

        Returns:
            Formatted text representation
        """
        parts = []

        # Role
        if 'role' in job:
            parts.append(f"Role: {job['role']}")

        # Company
        if 'company' in job:
            parts.append(f"Company: {job['company']}")

        # Skills
        if 'skills' in job:
            skills = job['skills']
            if isinstance(skills, list):
                parts.append(f"Skills: {', '.join(skills)}")
            else:
                parts.append(f"Skills: {skills}")

        # Description
        if 'description' in job:
            desc = job['description']
            # Truncate very long descriptions
            if len(desc) > 500:
                desc = desc[:500] + "..."
            parts.append(f"Description: {desc}")

        # Experience
        if 'experience' in job:
            parts.append(f"Experience: {job['experience']}")

        # Location
        if 'location' in job:
            parts.append(f"Location: {job['location']}")

        return "\n".join(parts)

    def search_matching_jobs(
        self,
        session_id: str,
        candidate_skills: List[str],
        candidate_experience: Optional[List[Dict]] = None,
        candidate_projects: Optional[List[Dict]] = None,
        n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find jobs matching candidate profile using semantic search.

        Args:
            session_id: Session ID to search within
            candidate_skills: List of candidate's skills
            candidate_experience: Optional candidate experience
            candidate_projects: Optional candidate projects
            n_results: Number of top results to return

        Returns:
            List of matching job dictionaries with similarity scores
        """
        logger.info(f"Searching for top {n_results} matching jobs for session: {session_id}")

        # Build query from candidate profile
        query = self._build_search_query(candidate_skills, candidate_experience, candidate_projects)

        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()

        # Search in ChromaDB (filter by session_id)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"session_id": session_id}
        )

        # Format results
        matching_jobs = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for idx in range(len(results['documents'][0])):
                job_data = {
                    'document': results['documents'][0][idx],
                    'metadata': results['metadatas'][0][idx],
                    'distance': results['distances'][0][idx] if 'distances' in results else None,
                    'similarity_score': self._distance_to_similarity(
                        results['distances'][0][idx]
                    ) if 'distances' in results else None
                }
                matching_jobs.append(job_data)

        logger.info(f"Found {len(matching_jobs)} matching jobs")
        return matching_jobs

    def _build_search_query(
        self,
        skills: List[str],
        experience: Optional[List[Dict]] = None,
        projects: Optional[List[Dict]] = None
    ) -> str:
        """
        Build search query from candidate profile.

        Args:
            skills: Candidate skills
            experience: Candidate experience
            projects: Candidate projects

        Returns:
            Formatted query string
        """
        query_parts = []

        # Skills
        if skills:
            query_parts.append(f"Skills: {', '.join(skills)}")

        # Experience
        if experience:
            exp_summaries = []
            for exp in experience[:3]:  # Top 3 experiences
                if 'role' in exp:
                    exp_summaries.append(exp['role'])
            if exp_summaries:
                query_parts.append(f"Experience: {', '.join(exp_summaries)}")

        # Projects
        if projects:
            project_techs = []
            for proj in projects[:3]:  # Top 3 projects
                if 'technologies_used' in proj:
                    techs = proj['technologies_used']
                    if isinstance(techs, list):
                        project_techs.extend(techs)
            if project_techs:
                query_parts.append(f"Technologies: {', '.join(set(project_techs))}")

        return "\n".join(query_parts)

    def _distance_to_similarity(self, distance: float) -> float:
        """
        Convert distance to similarity score (0-100%).

        Args:
            distance: Distance from ChromaDB (L2 distance)

        Returns:
            Similarity score between 0 and 100
        """
        # L2 distance -> similarity (inverse relationship)
        # Typical L2 distances range from 0 (identical) to ~2 (very different)
        # Convert to percentage
        similarity = max(0, min(100, (1 - distance / 2) * 100))
        return round(similarity, 2)

    def get_all_jobs_for_session(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all jobs for a given session (fallback if semantic search returns too few).

        Args:
            session_id: Session ID

        Returns:
            List of all jobs in session
        """
        results = self.collection.get(
            where={"session_id": session_id}
        )

        jobs = []
        if results['documents']:
            for idx in range(len(results['documents'])):
                job_data = {
                    'document': results['documents'][idx],
                    'metadata': results['metadatas'][idx]
                }
                jobs.append(job_data)

        return jobs

    def delete_session(self, session_id: str):
        """
        Delete all jobs for a given session.

        Args:
            session_id: Session ID to cleanup
        """
        try:
            # Get all IDs for this session
            results = self.collection.get(
                where={"session_id": session_id}
            )

            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} jobs for session: {session_id}")
            else:
                logger.info(f"No jobs found for session: {session_id}")

        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the job collection.

        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        return {
            "total_jobs": count,
            "collection_name": self.collection.name,
            "persist_directory": self.persist_directory
        }
