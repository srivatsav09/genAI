# Changelog

All notable changes to the Cold Email Generator project will be documented in this file.

## [2.0.0] - 2025-12-05

### Major Refactor and Feature Enhancements

#### Added
- **Advanced Skill Matching System** (`app/matcher.py`)
  - Fuzzy matching algorithm with configurable similarity threshold
  - Comprehensive scoring system for job matches
  - Detailed breakdown of required/desired skills matching
  - Experience and project relevance scoring
  - Overall match percentage calculation

- **DOCX Resume Support** (`app/utils.py`)
  - Extract text from Microsoft Word documents
  - Support for tables within DOCX files
  - Unified resume extraction interface for PDF and DOCX

- **Configuration Management** (`app/config.py`)
  - Centralized configuration with environment variables
  - Validation of required settings
  - Default values for all optional settings
  - Easy customization of LLM, UI, and matching parameters

- **Logging System** (`app/logger.py`)
  - Structured logging throughout the application
  - Configurable log levels
  - Optional file logging
  - Better error tracking and debugging

- **Enhanced UI Features**
  - Visual score breakdown with metrics
  - Detailed skill analysis in expandable sections
  - Progress indicators with step-by-step status
  - Improved error messages and validation
  - Download button for generated emails
  - Session state management for better UX

- **Email Customization**
  - Three tone options: professional, enthusiastic, casual
  - Tone-specific prompt engineering
  - User profile customization (name, university, year)
  - Better email structure and formatting

- **Docker Support**
  - Production-ready Dockerfile
  - Docker Compose configuration
  - Health checks and proper user permissions
  - Volume mounting for logs

- **Documentation**
  - Comprehensive README with examples
  - Architecture documentation
  - Docker deployment instructions
  - Troubleshooting guide
  - .env.example for easy setup

#### Changed
- **Complete Code Refactor**
  - Added type hints throughout entire codebase
  - Comprehensive docstrings for all functions and classes
  - Better error handling with specific exceptions
  - Separation of concerns (UI, business logic, utilities)
  - Consistent naming conventions

- **Improved Resume Parsing** (`app/resumeParser.py`)
  - Better extraction of skills, projects, and experience
  - Validation of parsed data
  - More detailed project information extraction
  - Duration parsing for experience entries

- **Enhanced Job Extraction** (`app/chain.py`)
  - Better prompt engineering for job extraction
  - Improved handling of multiple jobs
  - More structured job data format
  - Better error messages

- **Better Text Cleaning** (`app/utils.py`)
  - More robust HTML tag removal
  - URL extraction
  - Special character handling
  - Whitespace normalization

- **Updated Requirements** (`requirements.txt`)
  - Cleaned up dependencies
  - Removed unused packages
  - Added python-docx for DOCX support
  - Fixed encoding issues

#### Removed
- Portfolio CSV system (no longer needed for student use case)
- Commented-out legacy code
- Unused imports and variables
- ChromaDB dependency (vector database not needed)

### Technical Improvements

#### Code Quality
- 100% of functions have type hints
- Comprehensive error handling
- Logging added to all critical operations
- Input validation throughout
- Better exception messages

#### Performance
- Efficient skill matching with caching potential
- Optimized text extraction
- Reduced redundant LLM calls

#### Scalability
- Modular architecture for easy extensions
- Configuration-driven behavior
- Docker support for cloud deployment
- Logging for monitoring

### Breaking Changes
- Removed portfolio CSV functionality
- Changed import paths (now using `app` package)
- Environment variables moved to centralized config
- Resume parser API changed (now requires Chain instance)

### Migration Guide

If upgrading from v1.x:

1. **Create `.env` file** from `.env.example`
2. **Update imports** to use new package structure
3. **Remove portfolio CSV** references
4. **Install new dependencies**: `pip install -r requirements.txt`
5. **Update environment variables** to match new config system

### Files Changed
- `app/main.py` - Complete rewrite with better UI and session management
- `app/chain.py` - Added type hints, better prompts, tone support
- `app/resumeParser.py` - Improved parsing, validation
- `app/utils.py` - Added DOCX support, better utilities
- `requirements.txt` - Cleaned and updated
- `.gitignore` - Comprehensive ignore rules

### Files Added
- `app/__init__.py` - Package initialization
- `app/config.py` - Configuration management
- `app/logger.py` - Logging setup
- `app/matcher.py` - Advanced skill matching
- `Dockerfile` - Docker container definition
- `docker-compose.yml` - Docker Compose setup
- `.env.example` - Environment template
- `README.md` - Comprehensive documentation
- `CHANGELOG.md` - This file

### Files Removed
- `app/resources/my_portfolio.csv` - No longer needed

## [1.0.0] - 2025-04-14

### Initial Release
- Basic resume parsing from PDF
- Job extraction from career pages
- LLM-based job matching
- Cold email generation
- Streamlit UI
