"""
Streamlit application for generating personalized cold emails.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports (works from both root and app directory)
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

try:
    from app.chain import Chain
    from app.resumeParser import ResumeParser, validate_resume_data
    from app.job_rag import JobRAG
    from app.utils import (
        clean_text,
        extract_text_from_resume,
        format_job_for_display,
        validate_url
    )
    from app.config import Config
    from app.logger import setup_logger
    from app.matcher import SkillMatcher, rank_jobs
except ImportError:
    # Fallback if running from app directory
    from chain import Chain
    from resumeParser import ResumeParser, validate_resume_data
    from job_rag import JobRAG
    from utils import (
        clean_text,
        extract_text_from_resume,
        format_job_for_display,
        validate_url
    )
    from config import Config
    from logger import setup_logger
    from matcher import SkillMatcher, rank_jobs

logger = setup_logger(__name__)


def initialize_session_state():
    """Initialize session state variables."""
    if 'generated_email' not in st.session_state:
        st.session_state.generated_email = None
    if 'match_result' not in st.session_state:
        st.session_state.match_result = None
    if 'resume_data' not in st.session_state:
        st.session_state.resume_data = None
    if 'score_breakdown' not in st.session_state:
        st.session_state.score_breakdown = None


def render_header():
    """Render the application header."""
    st.title(f"{Config.APP_ICON} {Config.APP_TITLE}")
    st.markdown(
        """
        Upload your resume and provide a careers page URL to generate a personalized cold email
        tailored to the best matching job opportunity.
        """
    )
    st.markdown("---")


def render_sidebar(resume_parser):
    """Render the sidebar with settings and information."""
    with st.sidebar:
        st.header("Settings")

        # User information
        st.subheader("Your Information")
        user_name = st.text_input(
            "Name",
            value=Config.DEFAULT_USER_NAME,
            help="Your full name"
        )
        university = st.text_input(
            "University",
            value=Config.DEFAULT_UNIVERSITY,
            help="Your university/college name"
        )
        year = st.text_input(
            "Academic Year",
            value=Config.DEFAULT_YEAR,
            help="e.g., 4th year, Final year, etc."
        )

        # Email tone selection
        st.subheader("Email Tone")
        tone = st.selectbox(
            "Select tone",
            options=Config.EMAIL_TONES,
            index=Config.EMAIL_TONES.index(Config.DEFAULT_EMAIL_TONE),
            help="Choose the tone for your cold email"
        )

        st.markdown("---")

        # RAG Status
        st.subheader("🚀 RAG Features")
        if resume_parser.use_rag:
            st.success("✅ Resume RAG: Active")
            st.caption("Chunking resume for 60-80% faster parsing")
        else:
            st.info("ℹ️ Resume RAG: Disabled")

        st.success("✅ Job RAG: Active")
        st.caption("Semantic job matching for better results")

        st.markdown("---")

        # Information
        st.subheader("About")
        st.markdown(
            """
            This tool helps students:
            - Parse their resume automatically (with RAG!)
            - Match skills with job requirements
            - Generate personalized cold emails
            - Identify skill gaps for other opportunities
            """
        )

        return user_name, university, year, tone


def render_input_section():
    """Render the input section for resume and URL."""
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Upload Your Resume")
        uploaded_resume = st.file_uploader(
            "Choose your resume (PDF only)",
            type=Config.ALLOWED_RESUME_FORMATS,
            help=f"Maximum file size: {Config.MAX_UPLOAD_SIZE_MB}MB"
        )

        if uploaded_resume:
            st.success(f"Uploaded: {uploaded_resume.name}")

    with col2:
        st.subheader("2. Careers Page URL")
        url_input = st.text_input(
            "Enter the careers page URL",
            value="https://careers.nike.com/jobs",
            help="Paste the URL of the company's careers/jobs page"
        )

        if url_input and not validate_url(url_input):
            st.warning("Please enter a valid URL starting with http:// or https://")

    return uploaded_resume, url_input


def render_results(match_result: dict, email: str, tone: str, score_breakdown: dict = None):
    """Render the results section with job match and generated email."""

    # Best Match
    st.subheader("Best Job Match")
    best_match = match_result.get('best_match', {})

    with st.expander(f"📌 {best_match.get('role', 'Unknown Role')}", expanded=True):
        st.markdown(format_job_for_display(best_match))

    # Show score breakdown if available (outside the expander to avoid nesting)
    if score_breakdown:
        st.markdown("---")
        st.markdown("### Match Score Breakdown")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Overall Match",
                f"{score_breakdown.get('overall_score', 0):.1f}%"
            )
        with col2:
            req_match = score_breakdown.get('required_skills_match', {})
            st.metric(
                "Required Skills",
                f"{req_match.get('percentage', 0):.1f}%"
            )
        with col3:
            st.metric(
                "Experience Match",
                f"{score_breakdown.get('experience_score', 0):.1f}%"
            )

        # Detailed skill breakdown in a separate expander
        with st.expander("View Detailed Skill Analysis"):
            req_match = score_breakdown.get('required_skills_match', {})
            des_match = score_breakdown.get('desired_skills_match', {})

            if req_match.get('matched'):
                st.markdown("**✅ Matched Required Skills:**")
                for skill in req_match['matched']:
                    st.markdown(f"- {skill}")

            if req_match.get('missing'):
                st.markdown("\n**❌ Missing Required Skills:**")
                for skill in req_match['missing']:
                    st.markdown(f"- {skill}")

            if des_match.get('matched'):
                st.markdown("\n**✅ Matched Desired Skills:**")
                for skill in des_match['matched']:
                    st.markdown(f"- {skill}")

    # Reason for match
    st.subheader("Why This Job?")
    reason = match_result.get('reason', 'No explanation provided')
    st.info(reason)

    # Generated Email
    st.subheader(f"Generated Cold Email ({tone.capitalize()} Tone)")

    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("Copy the email below:")
    with col2:
        if st.button("📋 Copy Email"):
            st.toast("Email copied to clipboard!")

    st.text_area(
        "Your personalized email",
        value=email,
        height=400,
        label_visibility="collapsed"
    )

    # Download button
    st.download_button(
        label="📥 Download Email as Text",
        data=email,
        file_name=f"cold_email_{best_match.get('role', 'job').replace(' ', '_').lower()}.txt",
        mime="text/plain"
    )

    # Close matches / Skill gaps
    close_matches = match_result.get('close_matches', [])
    if close_matches:
        st.markdown("---")
        st.subheader("Other Potential Opportunities")
        st.markdown(
            "These roles are also good matches, but you're missing some skills. "
            "Here's how you can improve:"
        )

        for idx, match in enumerate(close_matches, start=1):
            with st.expander(f"Alternative {idx}: {match.get('role', 'Unknown Role')}"):
                st.markdown(f"**Missing Skills:**")
                missing_skills = match.get('missing_skills', [])
                for skill in missing_skills:
                    st.markdown(f"- {skill}")

                st.markdown(f"\n**Recommendation:**")
                st.info(match.get('recommendation', 'No recommendation provided'))


def process_application(
    uploaded_resume,
    url_input: str,
    user_name: str,
    university: str,
    year: str,
    tone: str,
    chain: Chain,
    resume_parser: ResumeParser,
    job_rag: JobRAG
):
    """Process the resume and generate cold email with RAG-based job matching."""

    # Validation
    if not uploaded_resume:
        st.warning("Please upload your resume first.")
        return False

    if not url_input or not validate_url(url_input):
        st.warning("Please enter a valid careers page URL.")
        return False

    progress_bar = st.progress(0)
    status_text = st.empty()
    job_session_id = None

    try:
        # Step 1: Extract resume text
        status_text.text("Step 1/6: Extracting text from resume...")
        progress_bar.progress(15)

        # Determine file type from filename
        file_extension = uploaded_resume.name.split('.')[-1].lower()
        resume_text = extract_text_from_resume(uploaded_resume, file_extension)
        logger.info(f"Extracted {len(resume_text)} characters from {file_extension.upper()} resume")

        # Step 2: Parse resume
        status_text.text("Step 2/6: Analyzing your resume with RAG...")
        progress_bar.progress(30)
        resume_data = resume_parser.parse_resume(chain, resume_text)

        if not validate_resume_data(resume_data):
            st.error("Failed to parse resume properly. Please ensure your resume is well-formatted.")
            return False

        st.session_state.resume_data = resume_data
        logger.info("Resume parsed successfully")

        # Step 3: Scrape jobs from URL
        status_text.text("Step 3/6: Fetching job listings from URL...")
        progress_bar.progress(45)
        loader = WebBaseLoader([url_input])
        raw_data = loader.load().pop().page_content
        cleaned_data = clean_text(raw_data)
        logger.info(f"Scraped and cleaned {len(cleaned_data)} characters from URL")

        # Step 4: Extract jobs and store in RAG
        status_text.text("Step 4/6: Extracting and indexing jobs with RAG...")
        progress_bar.progress(60)
        job_list = chain.extract_jobs(cleaned_data)

        if not job_list:
            st.warning("No job listings found on this page. Please try a different URL.")
            return False

        logger.info(f"Found {len(job_list)} job(s)")

        # Store jobs in JobRAG
        import uuid
        job_session_id = f"jobs_{uuid.uuid4().hex[:8]}"
        job_rag.store_jobs(job_list, job_session_id)
        logger.info(f"Stored {len(job_list)} jobs in RAG system")

        # Step 5: Semantic search for top matching jobs
        status_text.text("Step 5/6: Finding best matches using semantic search...")
        progress_bar.progress(75)

        # Use RAG to find top 10 matching jobs
        top_matching_jobs = job_rag.search_matching_jobs(
            session_id=job_session_id,
            candidate_skills=resume_data.get('skills', []),
            candidate_experience=resume_data.get('experience', []),
            candidate_projects=resume_data.get('projects', []),
            n_results=min(10, len(job_list))  # Top 10 or fewer
        )

        if not top_matching_jobs:
            st.warning("No matching jobs found. Try a different careers page.")
            return False

        logger.info(f"RAG found {len(top_matching_jobs)} matching jobs")

        # Convert RAG results back to job dictionaries for compatibility
        # We need to find the original jobs from job_list based on metadata
        matched_job_objects = []
        for rag_result in top_matching_jobs:
            job_index = rag_result['metadata'].get('job_index')
            if job_index is not None and job_index < len(job_list):
                matched_job_objects.append(job_list[job_index])

        # Use skill matcher to get detailed scores for top matches
        matcher = SkillMatcher(similarity_threshold=Config.MIN_SKILL_MATCH_THRESHOLD)
        ranked_jobs = rank_jobs(resume_data, matched_job_objects, matcher)

        # Get the best match score
        _, best_score = ranked_jobs[0]

        # Use LLM for comprehensive matching analysis on TOP matches only (not all jobs)
        match_result = chain.match_best_job(
            matched_job_objects,  # Only top RAG-matched jobs, not all jobs
            resume_data.get('skills', []),
            resume_data.get('projects', []),
            resume_data.get('experience', [])
        )
        st.session_state.match_result = match_result
        st.session_state.score_breakdown = best_score

        # Step 6: Generate email
        status_text.text("Step 6/6: Generating your personalized cold email...")
        progress_bar.progress(90)

        email = chain.write_mail(
            match_result['best_match'],
            resume_data.get('projects', []),
            resume_data.get('experience', []),
            tone=tone,
            user_name=user_name,
            university=university,
            year=year
        )
        st.session_state.generated_email = email

        progress_bar.progress(100)
        status_text.text("Done! Your cold email is ready.")
        logger.info("Successfully generated cold email")

        # Clean up RAG sessions
        if resume_data and '_rag_session_id' in resume_data:
            resume_parser.cleanup_session(resume_data['_rag_session_id'])
            logger.info("Cleaned up resume RAG session data")

        if job_session_id:
            job_rag.delete_session(job_session_id)
            logger.info("Cleaned up job RAG session data")

        return True

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)
        return False
    finally:
        progress_bar.empty()
        status_text.empty()


def main():
    """Main application function."""
    # Configure page
    st.set_page_config(
        layout=Config.PAGE_LAYOUT,
        page_title=Config.APP_TITLE,
        page_icon=Config.APP_ICON
    )

    # Initialize
    initialize_session_state()

    # Initialize components
    try:
        chain = Chain()
        resume_parser = ResumeParser()
        job_rag = JobRAG()
    except Exception as e:
        st.error(f"Failed to initialize application: {str(e)}")
        st.stop()

    # Render UI
    render_header()
    user_name, university, year, tone = render_sidebar(resume_parser)
    uploaded_resume, url_input = render_input_section()

    # Generate button
    st.markdown("---")
    _, col2, _ = st.columns([1, 1, 1])
    with col2:
        generate_button = st.button(
            "🚀 Generate Cold Email",
            type="primary",
            use_container_width=True
        )

    # Process
    if generate_button:
        success = process_application(
            uploaded_resume,
            url_input,
            user_name,
            university,
            year,
            tone,
            chain,
            resume_parser,
            job_rag
        )

        if success:
            st.success("Email generated successfully!")

    # Display results if available
    if st.session_state.generated_email and st.session_state.match_result:
        st.markdown("---")
        render_results(
            st.session_state.match_result,
            st.session_state.generated_email,
            tone,
            st.session_state.score_breakdown
        )

    # Footer
    st.markdown("---")
    st.caption("Made for students | Powered by LangChain + Groq + Streamlit")


if __name__ == "__main__":
    main()
