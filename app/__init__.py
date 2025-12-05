"""
Cold Email Generator Application Package.

A Streamlit application that helps students generate personalized cold emails
by matching their resume with job listings.
"""

__version__ = "2.0.0"
__author__ = "Srivatsav"

from app.chain import Chain
from app.resumeParser import ResumeParser
from app.config import Config

__all__ = ["Chain", "ResumeParser", "Config"]
