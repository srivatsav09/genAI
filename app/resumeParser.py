"""
Resume parsing functionality to extract skills, projects, and experience.
"""
from typing import Dict, Any, List, Optional
import uuid
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

from app.logger import setup_logger
from app.resume_rag import ResumeRAG

logger = setup_logger(__name__)


class ResumeParser:
    """Handles parsing of resume text to extract structured information."""

    def __init__(self, use_rag: bool = True):
        """
        Initialize the ResumeParser.

        Args:
            use_rag: Whether to use RAG for efficient resume processing (default: True)
        """
        self.use_rag = use_rag
        self.rag = None

        if self.use_rag:
            try:
                self.rag = ResumeRAG()
                logger.info("ResumeParser initialized with RAG enabled")
            except ImportError as e:
                logger.warning(f"RAG not available: {e}. Falling back to full resume processing")
                self.use_rag = False
                logger.info("ResumeParser initialized without RAG")
        else:
            logger.info("ResumeParser initialized without RAG")

    def parse_resume(self, llm, resume_text: str) -> Dict[str, Any]:
        """
        Parse resume text and extract structured information.
        Uses RAG for efficient processing if enabled.

        Args:
            llm: Language model instance (Chain object)
            resume_text: Raw text extracted from resume

        Returns:
            Dictionary containing:
                - skills: List of technical skills
                - projects: List of project dictionaries
                - experience: List of work experience dictionaries
                - session_id: RAG session ID (if RAG is used, for cleanup later)

        Raises:
            OutputParserException: If resume cannot be parsed properly
            ValueError: If resume text is empty or invalid
        """
        if not resume_text or not resume_text.strip():
            raise ValueError("Resume text cannot be empty")

        session_id = None
        context = resume_text

        # Use RAG if enabled
        if self.use_rag and self.rag:
            try:
                logger.info("Using RAG for resume processing")

                # Generate unique session ID
                session_id = f"resume_{uuid.uuid4().hex[:8]}"

                # Store resume in RAG system
                self.rag.store_resume(resume_text, session_id)

                # Retrieve relevant chunks for different queries
                skills_chunks = self.rag.retrieve_relevant_chunks(
                    session_id=session_id,
                    query="technical skills, programming languages, frameworks, tools, technologies",
                    n_results=2
                )

                projects_chunks = self.rag.retrieve_relevant_chunks(
                    session_id=session_id,
                    query="projects, portfolio work, applications built",
                    n_results=2
                )

                experience_chunks = self.rag.retrieve_relevant_chunks(
                    session_id=session_id,
                    query="work experience, internships, employment, professional experience",
                    n_results=2
                )

                # Combine all unique chunks
                all_chunks = skills_chunks + projects_chunks + experience_chunks
                unique_texts = []
                seen_texts = set()

                for chunk in all_chunks:
                    if chunk['text'] not in seen_texts:
                        unique_texts.append(chunk['text'])
                        seen_texts.add(chunk['text'])

                # Create focused context from retrieved chunks
                context = "\n\n".join(unique_texts)

                logger.info(f"RAG retrieved {len(unique_texts)} unique chunks (reduced from full resume)")

            except Exception as e:
                logger.warning(f"RAG processing failed: {e}. Falling back to full resume")
                context = resume_text
                session_id = None

        logger.info("Parsing resume text")

        prompt = PromptTemplate.from_template("""
        You are a helpful assistant specialized in analyzing student resumes.

        Here is a student's resume:
        {resume_text}

        Extract the following information:

        1. **Skills/Technologies/Tools**: List ALL technical skills, programming languages,
           frameworks, tools, platforms, and technologies mentioned anywhere in the resume.
           Include skills from projects, coursework, certifications, and experience sections.

        2. **Projects**: Extract 1-5 most relevant/impressive projects with:
           - Project title
           - Brief description (1-2 sentences)
           - Technologies/tools used
           - Key achievements or outcomes (if mentioned)

        3. **Experience**: Extract work experience, internships, or research positions with:
           - Company/Organization name
           - Role/Position title
           - Duration in months (estimate if not exact)
           - Key responsibilities or achievements (if mentioned)

        Return ONLY valid JSON in this exact format:
        {{
            "skills": ["skill1", "skill2", "skill3", ...],
            "projects": [
                {{
                    "title": "Project Name",
                    "description": "Brief description of the project",
                    "technologies_used": ["tech1", "tech2"],
                    "achievements": "Key outcomes or achievements"
                }}
            ],
            "experience": [
                {{
                    "company": "Company Name",
                    "role": "Position Title",
                    "duration_months": 6,
                    "responsibilities": "Key responsibilities and achievements"
                }}
            ]
        }}

        If any section is not found in the resume, return an empty array for that section.
        Ensure the JSON is properly formatted with no preamble or explanation.
        """)

        try:
            chain = prompt | llm.llm
            res = chain.invoke({"resume_text": context})  # Use context (RAG chunks or full resume)

            json_parser = JsonOutputParser()
            parsed_data = json_parser.parse(res.content)

            # Validate structure
            required_keys = ["skills", "projects", "experience"]
            for key in required_keys:
                if key not in parsed_data:
                    parsed_data[key] = []

            # Add session_id for cleanup later
            if session_id:
                parsed_data["_rag_session_id"] = session_id

            logger.info(
                f"Successfully parsed resume: "
                f"{len(parsed_data.get('skills', []))} skills, "
                f"{len(parsed_data.get('projects', []))} projects, "
                f"{len(parsed_data.get('experience', []))} experience entries"
            )

            return parsed_data

        except OutputParserException as e:
            logger.error(f"Failed to parse resume: {str(e)}")
            raise OutputParserException(
                "Could not parse resume properly. "
                "Please ensure the resume is well-formatted and try again."
            )
        except Exception as e:
            logger.error(f"Unexpected error parsing resume: {str(e)}")
            raise

    def cleanup_session(self, session_id: str):
        """
        Clean up RAG session data after processing.

        Args:
            session_id: RAG session ID to cleanup
        """
        if self.use_rag and self.rag and session_id:
            try:
                self.rag.delete_session(session_id)
                logger.info(f"Cleaned up RAG session: {session_id}")
            except Exception as e:
                logger.warning(f"Failed to cleanup RAG session {session_id}: {e}")


def validate_resume_data(data: Dict[str, Any]) -> bool:
    """
    Validate that parsed resume data has the required structure.

    Args:
        data: Parsed resume dictionary

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(data, dict):
        return False

    required_keys = ["skills", "projects", "experience"]
    for key in required_keys:
        if key not in data:
            return False
        if not isinstance(data[key], list):
            return False

    return True
