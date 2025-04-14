import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

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
    
    def write_mail(self,jobs,Projects,Experience):
        prompt_email = PromptTemplate.from_template(
        """
            ###JOB DESCRIPTION
            {job_description}
            ###INSTRUCTION
            You are Srivatsav, a computer science student studying at Vellore Institute of Technology, Chennai.
            You are looking for internships to increase your experience and at the same time contribute towards the company.
            your job is to write a cold email to to the hr dept regarding the job mentioned above and fulfilling their needs.
            Mention the projects that may play a major role in this particular position from the following project list:{pjt}
            Also mention the companies and roles you have worked at before from the follwing experience list to impress the HR:{exp}
            Remember, you are Srivatsav, 4th year student at VIT chennai.
            ###VALID JSON (NO PREAMBLE):
        """
        )

        chain_email = prompt_email | self.llm 
        res = chain_email.invoke(input={'job_description': str(jobs),'pjt':Projects, 'exp':Experience})
        return res.content
    
    def match_best_job(self, job_list, skills, projects,experience):
        prompt_match = PromptTemplate.from_template(
            """
            You are a helpful assistant.

            A student is looking for a job.
            These are the students **skills**: {skills}
            These are the students **projects**: {projects}
            These are the students **experience**: {experience}

            Below is a list of job postings:
            {job_list}

            ## TASKS:
            1. Identify the **best matching job** and return it in JSON format with:
            - `role`
            - `experience`
            - `skills`
            - `description`

            2. Provide a short **explanation** (100-150 words) on *why* this job was selected.

            3. Also return a list of up to 3 **alternative job roles** that are a close match, but where the student is currently missing some key skills. For each one, include:
            - Job role
            - Missing or weak skills
            - Suggestion for improvement

            ## RESPONSE FORMAT:
            {{
                "best_match": {{
                    "role": "...",
                    "experience": "...",
                    "skills": [...],
                    "description": "..."
                }},
                "reason": "Short paragraph explaining the match...",
                "close_matches": [
                    {{
                        "role": "...",
                        "missing_skills": [...],
                        "recommendation": "..."
                    }},
                    ...
                ]
            }}
            """
        )

        chain_match = prompt_match | self.llm
        res = chain_match.invoke({
            "skills": skills,
            "projects": projects,
            "experience":experience,
            "job_list": str(job_list)
        })

        try:
            parser = JsonOutputParser()
            return parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("⚠️ Could not parse match output. Try again with a different resume or job list.")

