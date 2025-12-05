"""
Advanced job matching logic with scoring algorithms.
"""
from typing import List, Dict, Any, Tuple
from difflib import SequenceMatcher
from app.logger import setup_logger

logger = setup_logger(__name__)


class SkillMatcher:
    """Handles skill matching and scoring between candidate and job requirements."""

    def __init__(self, similarity_threshold: float = 0.75):
        """
        Initialize the SkillMatcher.

        Args:
            similarity_threshold: Minimum similarity score (0-1) to consider skills as matching
        """
        self.similarity_threshold = similarity_threshold

    def calculate_similarity(self, skill1: str, skill2: str) -> float:
        """
        Calculate similarity between two skills using fuzzy matching.

        Args:
            skill1: First skill string
            skill2: Second skill string

        Returns:
            Similarity score between 0 and 1
        """
        skill1 = skill1.lower().strip()
        skill2 = skill2.lower().strip()

        # Exact match
        if skill1 == skill2:
            return 1.0

        # Check if one contains the other
        if skill1 in skill2 or skill2 in skill1:
            return 0.9

        # Use sequence matcher for fuzzy matching
        return SequenceMatcher(None, skill1, skill2).ratio()

    def match_skills(
        self,
        candidate_skills: List[str],
        required_skills: List[str]
    ) -> Tuple[List[str], List[str], float]:
        """
        Match candidate skills against required skills.

        Args:
            candidate_skills: List of candidate's skills
            required_skills: List of required skills for job

        Returns:
            Tuple of (matched_skills, missing_skills, match_percentage)
        """
        matched = []
        missing = []

        for req_skill in required_skills:
            found_match = False

            for cand_skill in candidate_skills:
                similarity = self.calculate_similarity(req_skill, cand_skill)

                if similarity >= self.similarity_threshold:
                    matched.append(req_skill)
                    found_match = True
                    break

            if not found_match:
                missing.append(req_skill)

        match_percentage = len(matched) / len(required_skills) if required_skills else 0.0

        return matched, missing, match_percentage

    def calculate_job_score(
        self,
        candidate_profile: Dict[str, Any],
        job: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate a comprehensive score for how well a candidate matches a job.

        Args:
            candidate_profile: Dictionary with skills, projects, experience
            job: Job dictionary with requirements

        Returns:
            Dictionary with detailed scoring breakdown
        """
        candidate_skills = candidate_profile.get('skills', [])
        candidate_projects = candidate_profile.get('projects', [])
        candidate_experience = candidate_profile.get('experience', [])

        # Extract job requirements
        job_skills = job.get('skills', {})
        required_skills = job_skills.get('required', []) if isinstance(job_skills, dict) else []
        desired_skills = job_skills.get('desired', []) if isinstance(job_skills, dict) else []

        # Match required skills
        matched_req, missing_req, req_percentage = self.match_skills(
            candidate_skills,
            required_skills
        )

        # Match desired skills
        matched_des, missing_des, des_percentage = self.match_skills(
            candidate_skills,
            desired_skills
        )

        # Calculate experience score (simple heuristic)
        experience_score = self.calculate_experience_score(
            candidate_experience,
            job.get('experience', '')
        )

        # Calculate project relevance score
        project_score = self.calculate_project_score(
            candidate_projects,
            required_skills + desired_skills
        )

        # Weighted overall score
        overall_score = (
            req_percentage * 0.50 +      # Required skills: 50%
            des_percentage * 0.20 +      # Desired skills: 20%
            experience_score * 0.15 +    # Experience: 15%
            project_score * 0.15         # Projects: 15%
        )

        return {
            'overall_score': round(overall_score * 100, 2),
            'required_skills_match': {
                'matched': matched_req,
                'missing': missing_req,
                'percentage': round(req_percentage * 100, 2)
            },
            'desired_skills_match': {
                'matched': matched_des,
                'missing': missing_des,
                'percentage': round(des_percentage * 100, 2)
            },
            'experience_score': round(experience_score * 100, 2),
            'project_relevance_score': round(project_score * 100, 2)
        }

    def calculate_experience_score(
        self,
        candidate_experience: List[Dict[str, Any]],
        job_experience_req: str
    ) -> float:
        """
        Calculate experience matching score.

        Args:
            candidate_experience: List of experience dictionaries
            job_experience_req: Job's experience requirement string

        Returns:
            Experience score between 0 and 1
        """
        if not candidate_experience:
            return 0.3  # Base score for students with no experience

        # Calculate total months of experience
        total_months = sum(
            exp.get('duration_months', 0)
            for exp in candidate_experience
        )

        # Parse job requirement (simple heuristic)
        job_req_lower = job_experience_req.lower()

        # Check for years mentioned
        if 'entry' in job_req_lower or 'junior' in job_req_lower or 'intern' in job_req_lower:
            required_months = 0
        elif '1 year' in job_req_lower or '1-2' in job_req_lower:
            required_months = 12
        elif '2' in job_req_lower:
            required_months = 24
        elif '3' in job_req_lower:
            required_months = 36
        else:
            required_months = 12  # Default assumption

        # Calculate score
        if total_months >= required_months:
            return 1.0
        elif total_months >= required_months * 0.5:
            return 0.8
        elif total_months > 0:
            return 0.5
        else:
            return 0.3  # Base score for no experience

    def calculate_project_score(
        self,
        candidate_projects: List[Dict[str, Any]],
        relevant_skills: List[str]
    ) -> float:
        """
        Calculate project relevance score based on technologies used.

        Args:
            candidate_projects: List of project dictionaries
            relevant_skills: List of skills relevant to the job

        Returns:
            Project relevance score between 0 and 1
        """
        if not candidate_projects or not relevant_skills:
            return 0.0

        total_matches = 0
        total_comparisons = 0

        for project in candidate_projects:
            project_techs = project.get('technologies_used', [])

            for tech in project_techs:
                for skill in relevant_skills:
                    total_comparisons += 1
                    similarity = self.calculate_similarity(tech, skill)
                    if similarity >= self.similarity_threshold:
                        total_matches += 1
                        break

        if total_comparisons == 0:
            return 0.0

        return min(total_matches / len(relevant_skills), 1.0)


def rank_jobs(
    candidate_profile: Dict[str, Any],
    jobs: List[Dict[str, Any]],
    matcher: SkillMatcher = None
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Rank jobs based on how well they match the candidate profile.

    Args:
        candidate_profile: Dictionary with candidate's skills, projects, experience
        jobs: List of job dictionaries
        matcher: Optional SkillMatcher instance

    Returns:
        List of tuples (job, score_breakdown) sorted by overall score descending
    """
    if matcher is None:
        matcher = SkillMatcher()

    logger.info(f"Ranking {len(jobs)} jobs for candidate")

    ranked = []
    for job in jobs:
        score = matcher.calculate_job_score(candidate_profile, job)
        ranked.append((job, score))

    # Sort by overall score descending
    ranked.sort(key=lambda x: x[1]['overall_score'], reverse=True)

    logger.info(f"Top match: {ranked[0][0].get('role', 'Unknown')} with score {ranked[0][1]['overall_score']}")

    return ranked
