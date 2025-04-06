import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chain import Chain
from portfolio import Portfolio
from utils import clean_text


def create_streamlit_app(llm, portfolio, clean_text):
    st.title("📧 Cold Mail Generator")
    st.markdown(
        """
        Generate personalized cold emails from job postings using AI.
        Just paste a job URL below and let the app do the magic. ✨
        """
    )

    with st.container():
        st.subheader("🔗 Input Job Post URL")
        url_input = st.text_input("Enter a job post or company URL:", 
                                  value="https://careers.nike.com/cdn-security-engineer-waf/job/R-48004", 
                                  help="Paste the full URL to a job listing or career page.")
        submit_button = st.button("🚀 Generate Email")

    if submit_button:
        st.info("⏳ Processing... Please wait a few seconds.")
        try:
            loader = WebBaseLoader([url_input])
            data = clean_text(loader.load().pop().page_content)

            # Load portfolio and generate email
            portfolio.load_portfolio()
            jobs = llm.extract_jobs(data)

            if not jobs:
                st.warning("⚠️ No jobs found in the content. Please check the URL.")
                return

            st.subheader("📬 Generated Cold Emails")
            for i, job in enumerate(jobs, start=1):
                with st.expander(f"Email #{i}: {job.get('title', 'Untitled Role')}"):
                    skills = job.get('skills', [])
                    links = portfolio.query_links(skills['required'])
                    email = llm.write_mail(job, links)
                    st.code(email.strip(), language='markdown')

        except Exception as e:
            st.error("❌ An error occurred while generating the email:")
            st.exception(e)

    st.markdown("---")
    st.caption("Made with ❤️ using LangChain + Streamlit")


if __name__ == "__main__":
    chain = Chain()
    portfolio = Portfolio()
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    create_streamlit_app(chain, portfolio, clean_text)
