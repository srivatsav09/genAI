# import pandas as pd
# import chromadb
# import uuid


# class Portfolio:
#     def __init__(self, file_path="app/resources/my_portfolio.csv"):
#         self.file_path = file_path
#         self.data = pd.read_csv(file_path)
#         self.chroma_client = chromadb.PersistentClient('vectorstore')
#         self.collection = self.chroma_client.get_or_create_collection(name="portfolio")

#     def load_portfolio(self):
#         if not self.collection.count():
#             for _, row in self.data.iterrows():
#                 self.collection.add(documents=row["Techstack"],
#                                     metadatas={"links": row["Links"]},
#                                     ids=[str(uuid.uuid4())])

#     def query_links(self, skills):
#         return self.collection.query(query_texts=skills, n_results=2).get('metadatas', [])

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException


class ResumeParser:
    def __init__(self):
        pass  # No setup needed

    def parse_resume(self, llm, resume_text):
        prompt = PromptTemplate.from_template("""
        You are a helpful assistant.

        Here is a student's resume:
        {resume_text}

        Extract:
        - A list of **skills/technologies/tools** the student knows or has worked with (even if not explicitly listed as skills).
        - A summary of 1-3 most relevant **projects** including tools used.
        - The students **experience** in companies and for how long

        Return in the following JSON format:
        {{
            "skills": [...],
            "projects": [
                {{
                    "title": "...",
                    "description": "...",
                    "technologies_used": [...]
                }}
            ]
            "experience":[
                {{
                    "Company name":"...",
                    "Role":"...",
                    "Duration(in months)":"...
                }}
            ]
        }}
        """)
        try:
            chain = prompt | llm.llm
            res = chain.invoke({"resume_text": resume_text})
            json_parser = JsonOutputParser()
            return json_parser.parse(res.content)
        except OutputParserException as e:
            raise OutputParserException("Could not parse resume properly. Please try again.")