"""
LLM chain operations for job extraction, matching, and email generation.
"""
from typing import List, Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

from app.config import Config
from app.logger import setup_logger

logger = setup_logger(__name__)


class Chain:
    """Handles all LLM chain operations for the cold email generator."""

    def __init__(self, temperature: Optional[float] = None, model: Optional[str] = None):
        """
        Initialize the Chain with LLM configuration.

        Args:
            temperature: LLM temperature (0-1). Defaults to config value.
            model: LLM model name. Defaults to config value.
        """
        self.temperature = temperature if temperature is not None else Config.LLM_TEMPERATURE
        self.model = model or Config.LLM_MODEL

        try:
            self.llm = ChatGroq(
                temperature=self.temperature,
                groq_api_key=Config.GROQ_API_KEY,
                model=self.model
            )
            logger.info(f"Initialized LLM with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise

    def extract_jobs(self, cleaned_text: str) -> List[Dict[str, Any]]:
        """
        Extract job postings from cleaned webpage text.

        Args:
            cleaned_text: Cleaned text from careers page

        Returns:
            List of job dictionaries with keys: role, experience, skills, description

        Raises:
            OutputParserException: If the response cannot be parsed
            ValueError: If no jobs are found
        """
        logger.info("Extracting jobs from webpage text")

        prompt_extract = PromptTemplate.from_template(
            """
            ###SCRAPED DATA FROM WEBSITE
            {page_data}

            ###INSTRUCTION
            The scraped text is from the career's page of a website.
            Your job is to extract ALL job postings and return them in JSON format.

            For each job, include these keys:
            - `role`: Job title/position name
            - `experience`: Required years or level of experience
            - `skills`: Object with `required` and `desired` skill arrays
            - `description`: Brief job summary and key responsibilities

            If multiple jobs are present, return an array of job objects.
            If only one job, return a single job object.
            Only return valid JSON, no preamble or explanation.

            ###VALID JSON (NO PREAMBLE):
            """
        )

        try:
            chain_extract = prompt_extract | self.llm
            res = chain_extract.invoke(input={'page_data': cleaned_text})

            json_parser = JsonOutputParser()
            parsed_res = json_parser.parse(res.content)

            # Ensure we always return a list
            jobs = parsed_res if isinstance(parsed_res, list) else [parsed_res]

            logger.info(f"Successfully extracted {len(jobs)} job(s)")
            return jobs

        except OutputParserException as e:
            logger.error(f"Failed to parse job extraction output: {str(e)}")
            raise OutputParserException(
                "Could not parse jobs from the webpage. "
                "The content might be too large or not in the expected format."
            )
        except Exception as e:
            logger.error(f"Unexpected error during job extraction: {str(e)}")
            raise

    def match_best_job(
        self,
        job_list: List[Dict[str, Any]],
        skills: List[str],
        projects: List[Dict[str, Any]],
        experience: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Match the best job from a list based on student's profile.

        Args:
            job_list: List of job dictionaries
            skills: List of student's skills
            projects: List of student's projects
            experience: List of student's work experience

        Returns:
            Dictionary with:
                - best_match: The best matching job
                - reason: Explanation for the match
                - close_matches: Alternative jobs with missing skills

        Raises:
            OutputParserException: If the response cannot be parsed
        """
        logger.info(f"Matching best job from {len(job_list)} available jobs")

        prompt_match = PromptTemplate.from_template(
            """
            You are a helpful career advisor assistant.

            A student is looking for a job opportunity.

            **Student Profile:**
            - Skills: {skills}
            - Projects: {projects}
            - Experience: {experience}

            **Available Job Postings:**
            {job_list}

            ## YOUR TASKS:

            1. Identify the **best matching job** based on:
               - Skill alignment (required and desired skills)
               - Experience level match
               - Relevant projects
               Return the complete job object in JSON format with keys:
               `role`, `experience`, `skills`, `description`

            2. Provide a **clear explanation** (100-150 words) explaining:
               - Why this job is the best match
               - Which skills align well
               - How their projects/experience are relevant
               - What makes them a strong candidate

            3. List up to {max_matches} **alternative jobs** where the student is close but missing some skills:
               For each alternative, include:
               - Job role name
               - List of missing or weak skills
               - Specific recommendation for improvement

            ## RESPONSE FORMAT (valid JSON only):
            {{
                "best_match": {{
                    "role": "...",
                    "experience": "...",
                    "skills": {{"required": [...], "desired": [...]}},
                    "description": "..."
                }},
                "reason": "Detailed explanation paragraph...",
                "close_matches": [
                    {{
                        "role": "Job Title",
                        "missing_skills": ["skill1", "skill2"],
                        "recommendation": "Specific advice..."
                    }}
                ]
            }}
            """
        )

        try:
            chain_match = prompt_match | self.llm
            res = chain_match.invoke({
                "skills": skills,
                "projects": projects,
                "experience": experience,
                "job_list": str(job_list),
                "max_matches": Config.MAX_CLOSE_MATCHES
            })

            parser = JsonOutputParser()
            result = parser.parse(res.content)

            logger.info(f"Successfully matched best job: {result.get('best_match', {}).get('role', 'Unknown')}")
            return result

        except OutputParserException as e:
            logger.error(f"Failed to parse job matching output: {str(e)}")
            raise OutputParserException(
                "Could not parse the job matching results. "
                "Please try again with a different resume or job listing."
            )
        except Exception as e:
            logger.error(f"Unexpected error during job matching: {str(e)}")
            raise

    def write_mail(
        self,
        job: Dict[str, Any],
        projects: List[Dict[str, Any]],
        experience: List[Dict[str, Any]],
        tone: str = "professional",
        user_name: Optional[str] = None,
        university: Optional[str] = None,
        year: Optional[str] = None
    ) -> str:
        """
        Generate a personalized cold email for a job.

        Args:
            job: Job dictionary with role, skills, description
            projects: List of relevant student projects
            experience: List of student's work experience
            tone: Email tone (professional, enthusiastic, casual)
            user_name: Student's name (defaults to config)
            university: University name (defaults to config)
            year: Academic year (defaults to config)

        Returns:
            Generated email as string

        Raises:
            ValueError: If tone is not valid
        """
        if tone not in Config.EMAIL_TONES:
            raise ValueError(f"Invalid tone. Must be one of: {Config.EMAIL_TONES}")

        # Use defaults from config if not provided
        user_name = user_name or Config.DEFAULT_USER_NAME
        university = university or Config.DEFAULT_UNIVERSITY
        year = year or Config.DEFAULT_YEAR

        logger.info(f"Generating {tone} email for job: {job.get('role', 'Unknown')}")

        # Tone-specific instructions
        tone_instructions = {
            "professional": "Write in a formal, professional tone. Be respectful and concise.",
            "enthusiastic": "Write with enthusiasm and passion. Show genuine excitement about the opportunity.",
            "casual": "Write in a friendly, approachable tone while maintaining professionalism."
        }

        prompt_email = PromptTemplate.from_template(
            """
            ###JOB DESCRIPTION
            {job_description}

            ###STUDENT PROFILE
            Name: {user_name}
            University: {university}
            Year: {year}

            Relevant Projects: {projects}
            Work Experience: {experience}

            ###INSTRUCTION
            You are {user_name}, a {year} computer science student at {university}.
            You are looking for internship/job opportunities to gain experience and contribute to the company.

            Write a cold email to the hiring manager/HR department for the job mentioned above.

            **Tone**: {tone_instruction}

            **Email Requirements:**
            - Professional subject line
            - Clear introduction stating your interest
            - Highlight 2-3 most relevant projects that align with the job requirements
            - Mention relevant work experience if applicable
            - Explain why you're a good fit for this specific role
            - Show understanding of the company/role
            - Include a clear call-to-action
            - Keep it concise (250-350 words)
            - Professional closing

            **DO NOT:**
            - Use generic templates
            - Include placeholder links
            - Make it too long or too short
            - Sound desperate or overly humble

            Write the complete email below (including subject line):
            """
        )

        try:
            chain_email = prompt_email | self.llm
            res = chain_email.invoke({
                'job_description': str(job),
                'projects': str(projects),
                'experience': str(experience),
                'user_name': user_name,
                'university': university,
                'year': year,
                'tone_instruction': tone_instructions[tone]
            })

            email_content = res.content.strip()
            logger.info("Successfully generated email")
            return email_content

        except Exception as e:
            logger.error(f"Failed to generate email: {str(e)}")
            raise
