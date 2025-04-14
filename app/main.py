# import streamlit as st
# from langchain_community.document_loaders import WebBaseLoader

# from chain import Chain
# from app.ResumeParser import Portfolio
# from utils import clean_text


# def create_streamlit_app(llm, portfolio, clean_text):
#     st.title("📧 Cold Mail Generator")
#     st.markdown(
#         """
#         Generate personalized cold emails from job postings using AI.
#         Just paste a job URL below and let the app do the magic. ✨
#         """
#     )

#     with st.container():
#         st.subheader("🔗 Input Job Post URL")
#         url_input = st.text_input("Enter a job post or company URL:", 
#                                   value="https://careers.nike.com/cdn-security-engineer-waf/job/R-48004", 
#                                   help="Paste the full URL to a job listing or career page.")
#         submit_button = st.button("🚀 Generate Email")

#     if submit_button:
#         st.info("⏳ Processing... Please wait a few seconds.")
#         try:
#             loader = WebBaseLoader([url_input])
#             data = clean_text(loader.load().pop().page_content)

#             # Load portfolio and generate email
#             portfolio.load_portfolio()
#             jobs = llm.extract_jobs(data)

#             if not jobs:
#                 st.warning("⚠️ No jobs found in the content. Please check the URL.")
#                 return

#             st.subheader("📬 Generated Cold Emails")
#             for i, job in enumerate(jobs, start=1):
#                 with st.expander(f"Email #{i}: {job.get('title', 'Untitled Role')}"):
#                     skills = job.get('skills', [])
#                     links = portfolio.query_links(skills['required'])
#                     email = llm.write_mail(job, links)
#                     st.code(email.strip(), language='markdown')

#         except Exception as e:
#             st.error("❌ An error occurred while generating the email:")
#             st.exception(e)

#     st.markdown("---")
#     st.caption("Made with ❤️ using LangChain + Streamlit")


# if __name__ == "__main__":
#     chain = Chain()
#     portfolio = Portfolio()
#     st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
#     create_streamlit_app(chain, portfolio, clean_text)

import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from chain import Chain
from resumeParser import ResumeParser  # renamed from portfolio.py
from utils import clean_text
from PyPDF2 import PdfReader

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    return " ".join(page.extract_text() for page in reader.pages if page.extract_text())

def create_streamlit_app(llm, resume_parser, clean_text):
    st.title("🎓 Student Resume → Cold Mail Generator")
    st.markdown("Upload your resume and paste a job listing page to generate a personalized cold email!")

    with st.container():
        uploaded_resume = st.file_uploader("📄 Upload your resume (PDF only)", type=["pdf"])
        url_input = st.text_input("🔗 Paste the careers page URL", 
                                  value="https://careers.nike.com/jobs",
                                  help="This should be a page with multiple jobs listed.")
        submit_button = st.button("🚀 Generate Email")

    if submit_button:
        if not uploaded_resume:
            st.warning("⚠️ Please upload your resume first.")
            return

        st.info("⏳ Processing your resume and job listings...")

        try:
            # Step 1: Parse resume
            resume_text = extract_text_from_pdf(uploaded_resume)
            resume_data = resume_parser.parse_resume(llm, resume_text)
            skills = resume_data["skills"]
            projects = resume_data["projects"]
            experience = resume_data["experience"]

            # Step 2: Scrape and clean job listings
            loader = WebBaseLoader([url_input])
            raw_data = loader.load().pop().page_content
            cleaned_data = clean_text(raw_data)
            job_list = llm.extract_jobs(cleaned_data)

            # Step 3: Match best job
            match_result = llm.match_best_job(job_list, skills, projects,experience)
            matched_job = match_result["best_match"]
            reason = match_result["reason"]
            close_matches = match_result["close_matches"]

            email = llm.write_mail(matched_job, projects, experience)

            st.subheader("📬 Your Personalized Cold Email")
            st.code(email.strip(), language='markdown')

            st.markdown("### 🤔 Why This Job?")
            st.success(reason)

            if close_matches:
                st.markdown("### 🧠 Other Potential Matches (And What You’re Missing)")
                for match in close_matches:
                    st.markdown(f"""
                    **🧪 Role:** `{match["role"]}`  
                    **🚫 Missing Skills:** {", ".join(match["missing_skills"])}  
                    **📈 Tip:** {match["recommendation"]}
                    """)

        except Exception as e:
            st.error("❌ An error occurred:")
            st.exception(e)

    st.markdown("---")
    st.caption("Made for students with ❤️ using LangChain + Streamlit")


if __name__ == "__main__":
    chain = Chain()
    resume_parser = ResumeParser()
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    create_streamlit_app(chain, resume_parser, clean_text)