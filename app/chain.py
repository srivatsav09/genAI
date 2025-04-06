import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv('secret.env')

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0,groq_api_key = os.getenv("API_KEY"),model="llama-3.1-8b-instant")

    def extract_jobs(self,cleaned_text):
        prompt_extract = PromptTemplate.from_template(
        """
            ###SCRAPED DATA FROM WEBSITE
            {page_data}
            ###INSTRUCTION
            The scraped text is from the career's page of a website
            Your job is to extract the job postings and return them in JSON format containing
            following keys: `role`, `experience`, `skills`, and `description`.
            only return the valid JSON.
            ###VALID JSON (NO PREAMBLE):
        """
        )
        chain_extract = prompt_extract | self.llm 
        res = chain_extract.invoke(input={'page_data': cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("context too big. Unable to parse jobs")
        return res if isinstance(res,list) else [res]
    
    def write_mail(self,jobs,Links):
        prompt_email = PromptTemplate.from_template(
        """
            ###JOB DESCRIPTION
            {job_description}
            ###INSTRUCTION
            You are Srivatsav, a computer science student studying at Vellore Institute of Technology, Chennai.
            You are looking for internships to increase your experience and at the same time contribute towards the company.
            your job is to write a cold email to to the hr dept regarding the job mentioned above and fulfilling their needs.
            Also add the relevant ones from the follwing links to show your portfolio:{link_list}
            Remember, you are Srivatsav, 4th year student at VIT chennai.
            ###VALID JSON (NO PREAMBLE):
        """
        )

        chain_email = prompt_email | self.llm 
        res = chain_email.invoke(input={'job_description': str(jobs), 'link_list':Links})
        return res.content
