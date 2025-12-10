"""
Company Intelligence RAG system for H1B sponsorship and company data.
Stores company profiles, H1B history, and enables semantic search for company matching.
"""
from typing import List, Dict, Any, Optional
import uuid

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


class CompanyRAG:
    """
    RAG system for company intelligence with H1B sponsorship data.

    Enables semantic search for companies based on:
    - H1B sponsorship history
    - Job roles offered
    - Company metadata (industry, size, location)
    - Hiring patterns for international students
    """

    def __init__(self, persist_directory: str = "./vectorstore_companies"):
        """
        Initialize CompanyRAG system.

        Args:
            persist_directory: Directory to persist the vector database

        Raises:
            ImportError: If required dependencies are not installed
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb is required. Install with: pip install chromadb")

        if not EMBEDDINGS_AVAILABLE:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")

        logger.info("Initializing Company RAG system")

        # Initialize ChromaDB client
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Loaded embedding model: all-MiniLM-L6-v2 ({embedding_dim} dimensions)")

        # Create or get collection for companies
        self.collection = self.client.get_or_create_collection(
            name="company_intelligence",
            metadata={"description": "Company profiles with H1B and hiring data"}
        )

        logger.info("Company RAG system initialized")

    def store_company(self, company_data: Dict[str, Any]) -> str:
        """
        Store a single company profile in the vector database.

        Args:
            company_data: Dictionary containing company information with keys:
                - company_name (required): Company name
                - h1b_filings (optional): Number of H1B filings
                - roles_offered (optional): List of job roles
                - avg_salary (optional): Average salary for H1B roles
                - locations (optional): List of office locations
                - years_active (optional): Years with H1B filings

        Returns:
            Company ID (unique identifier)
        """
        if 'company_name' not in company_data:
            raise ValueError("company_name is required in company_data")

        company_name = company_data['company_name']
        logger.info(f"Storing company: {company_name}")

        # Create searchable text representation
        company_text = self._format_company_for_storage(company_data)

        # Generate embedding
        embedding = self.embedding_model.encode(company_text).tolist()

        # Create metadata
        metadata = {
            "company_name": company_name,
            "h1b_sponsor": True,  # All companies in our DB sponsor H1B
        }

        # Add optional metadata
        if 'h1b_filings' in company_data:
            metadata['h1b_filings'] = int(company_data['h1b_filings'])
        if 'avg_salary' in company_data:
            metadata['avg_salary'] = float(company_data['avg_salary'])

        # Generate unique ID
        company_id = f"company_{uuid.uuid4().hex[:8]}"

        # Store in ChromaDB
        self.collection.add(
            embeddings=[embedding],
            documents=[company_text],
            metadatas=[metadata],
            ids=[company_id]
        )

        logger.info(f"Successfully stored company: {company_name} (ID: {company_id})")
        return company_id

    def store_companies_bulk(self, companies: List[Dict[str, Any]]) -> int:
        """
        Store multiple companies in bulk.

        Args:
            companies: List of company dictionaries

        Returns:
            Number of companies stored
        """
        logger.info(f"Bulk storing {len(companies)} companies")

        documents = []
        embeddings_list = []
        metadatas = []
        ids = []

        for company_data in companies:
            if 'company_name' not in company_data:
                logger.warning(f"Skipping company without name: {company_data}")
                continue

            # Create searchable text
            company_text = self._format_company_for_storage(company_data)
            documents.append(company_text)

            # Create metadata
            metadata = {
                "company_name": company_data['company_name'],
                "h1b_sponsor": True,
            }

            if 'h1b_filings' in company_data:
                metadata['h1b_filings'] = int(company_data['h1b_filings'])
            if 'avg_salary' in company_data:
                metadata['avg_salary'] = float(company_data['avg_salary'])

            metadatas.append(metadata)

            # Generate unique ID
            company_id = f"company_{uuid.uuid4().hex[:8]}"
            ids.append(company_id)

        # Generate embeddings in batch
        embeddings_list = self.embedding_model.encode(documents).tolist()

        # Store in ChromaDB
        self.collection.add(
            embeddings=embeddings_list,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        logger.info(f"Successfully stored {len(ids)} companies")
        return len(ids)

    def _format_company_for_storage(self, company_data: Dict[str, Any]) -> str:
        """
        Format company data into searchable text.

        Args:
            company_data: Company dictionary

        Returns:
            Formatted text representation
        """
        parts = []

        # Company name
        parts.append(f"Company: {company_data['company_name']}")
        parts.append("H1B Sponsor: Yes")

        # H1B filings
        if 'h1b_filings' in company_data:
            parts.append(f"H1B Filings: {company_data['h1b_filings']}")

        # Roles offered
        if 'roles_offered' in company_data:
            roles = company_data['roles_offered']
            if isinstance(roles, list):
                parts.append(f"Roles: {', '.join(roles[:10])}")  # Top 10 roles
            else:
                parts.append(f"Roles: {roles}")

        # Average salary
        if 'avg_salary' in company_data:
            parts.append(f"Average H1B Salary: ${company_data['avg_salary']:,.0f}")

        # Locations
        if 'locations' in company_data:
            locations = company_data['locations']
            if isinstance(locations, list):
                parts.append(f"Locations: {', '.join(locations[:5])}")  # Top 5 locations
            else:
                parts.append(f"Locations: {locations}")

        # Years active
        if 'years_active' in company_data:
            years = company_data['years_active']
            if isinstance(years, list):
                parts.append(f"H1B Years: {min(years)}-{max(years)}")

        return "\n".join(parts)

    def search_companies(
        self,
        query: str,
        n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for H1B sponsoring companies matching query.

        Args:
            query: Search query (e.g., "Software Engineer H1B sponsor in California")
            n_results: Number of results to return

        Returns:
            List of matching companies with metadata and similarity scores
        """
        logger.info(f"Searching companies: '{query}' (n={n_results})")

        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()

        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        # Format results
        companies = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for idx in range(len(results['documents'][0])):
                company_data = {
                    'document': results['documents'][0][idx],
                    'metadata': results['metadatas'][0][idx],
                    'distance': results['distances'][0][idx] if 'distances' in results else None,
                    'similarity_score': self._distance_to_similarity(
                        results['distances'][0][idx]
                    ) if 'distances' in results else None
                }
                companies.append(company_data)

        logger.info(f"Found {len(companies)} matching companies")
        return companies

    def get_h1b_sponsors_for_role(self, role: str, n_results: int = 20) -> List[Dict[str, Any]]:
        """
        Get companies that sponsor H1B for a specific role.

        Args:
            role: Job role (e.g., "Software Engineer", "Data Scientist")
            n_results: Number of companies to return

        Returns:
            List of H1B sponsoring companies for the role
        """
        query = f"H1B sponsor for {role} position"
        return self.search_companies(query, n_results=n_results)

    def _distance_to_similarity(self, distance: float) -> float:
        """
        Convert distance to similarity score (0-100%).

        Args:
            distance: Distance from ChromaDB (L2 distance)

        Returns:
            Similarity score between 0 and 100
        """
        similarity = max(0, min(100, (1 - distance / 2) * 100))
        return round(similarity, 2)

    def get_company_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the company database.

        Returns:
            Dictionary with database statistics
        """
        total_companies = self.collection.count()

        return {
            "total_companies": total_companies,
            "collection_name": self.collection.name,
            "persist_directory": self.persist_directory
        }

    def delete_all_companies(self):
        """Delete all companies from the database. USE WITH CAUTION!"""
        logger.warning("Deleting all companies from database")
        self.client.delete_collection(name="company_intelligence")
        self.collection = self.client.get_or_create_collection(
            name="company_intelligence",
            metadata={"description": "Company profiles with H1B and hiring data"}
        )
        logger.info("All companies deleted, collection recreated")
