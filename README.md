# Cold Email Generator for Students

A powerful Streamlit application that helps students generate personalized cold emails by automatically matching their resume with job listings from company career pages.

## Features

### Core Features
- **Automatic Resume Parsing**: Extracts skills, projects, and experience from PDF and DOCX resumes
- **Advanced Skill Matching**: Uses fuzzy matching algorithms to calculate similarity scores
- **Smart Job Matching**: Analyzes multiple job listings and finds the best match based on your profile
- **Detailed Match Scoring**: Shows percentage matches for required/desired skills, experience, and projects
- **Personalized Email Generation**: Creates tailored cold emails with customizable tone
- **Skill Gap Analysis**: Identifies missing skills for alternative opportunities

### Customization
- **Multiple Email Tones**: Choose between professional, enthusiastic, or casual tones
- **User Profile**: Customize name, university, and academic year
- **Flexible Configuration**: Environment-based settings for easy customization

### Export & Analysis
- **Download & Export**: Save generated emails as text files
- **Visual Score Breakdown**: See detailed match scores with interactive UI
- **Alternative Job Suggestions**: Get recommendations for improving your skillset

## Architecture

```
genAI/
├── app/
│   ├── __init__.py           # Package initialization
│   ├── main.py               # Streamlit UI and main app logic
│   ├── chain.py              # LLM chains for job extraction, matching, and email generation
│   ├── matcher.py            # Advanced skill matching and scoring algorithms
│   ├── resumeParser.py       # Resume parsing and data extraction
│   ├── utils.py              # Utility functions (PDF/DOCX extraction, text cleaning)
│   ├── config.py             # Centralized configuration management
│   └── logger.py             # Logging configuration
├── .env                      # Environment variables (not in git)
├── .env.example              # Example environment configuration
├── .gitignore                # Git ignore rules
├── Dockerfile                # Docker container definition
├── docker-compose.yml        # Docker Compose configuration
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Groq API key ([Get one here](https://console.groq.com/))

### Setup

1. **Clone the repository**
   ```bash
   cd genAI
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv

   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   # Copy the example env file
   cp .env.example .env

   # Edit .env and add your Groq API key
   API_KEY=your_groq_api_key_here
   ```

## Usage

### Running the Application

#### Option 1: Using the Run Script (Recommended)
```bash
# Make sure you're in the genAI directory and venv is activated
cd genAI
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On macOS/Linux

python run.py
```

#### Option 2: Direct Streamlit Command
```bash
# Make sure you're in the genAI directory and venv is activated
cd genAI
venv\Scripts\activate  # On Windows

streamlit run app/main.py
```

The application will open in your default web browser at `http://localhost:8501`

**Note:** Make sure you have activated the virtual environment and your `.env` file is in the project root with your API key.

### Running with Docker

If you prefer to use Docker:

1. **Build and run using Docker Compose**
   ```bash
   # Make sure you have .env file with your API key
   docker-compose up -d
   ```

2. **Or build and run manually**
   ```bash
   # Build the image
   docker build -t cold-email-generator .

   # Run the container
   docker run -p 8501:8501 --env-file .env cold-email-generator
   ```

3. **Access the application**
   - Open your browser to `http://localhost:8501`

4. **Stop the application**
   ```bash
   docker-compose down
   ```

### How to Use

1. **Configure Your Information** (in sidebar):
   - Enter your name
   - Enter your university
   - Specify your academic year
   - Choose email tone (professional/enthusiastic/casual)

2. **Upload Your Resume**:
   - Click "Upload Your Resume"
   - Select your PDF resume file
   - The app will automatically extract your skills, projects, and experience

3. **Enter Careers Page URL**:
   - Paste the URL of a company's careers/jobs page
   - Example: `https://careers.nike.com/jobs`

4. **Generate Email**:
   - Click "Generate Cold Email"
   - Wait for the app to process (usually 20-30 seconds)
   - Review your personalized email

5. **Download & Use**:
   - Copy the email or download as a text file
   - Review the skill gap analysis for other opportunities
   - Customize the email if needed before sending

## Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# Required
API_KEY=your_groq_api_key_here

# Optional (with defaults)
LLM_MODEL=llama-3.1-8b-instant
LLM_TEMPERATURE=0
USER_NAME=Your Name
UNIVERSITY=Your University
YEAR=4th year
ENABLE_CACHE=true
CACHE_TTL_SECONDS=3600
LOG_LEVEL=INFO
```

### Customization

You can customize various aspects in `app/config.py`:
- Email tones
- Job matching thresholds
- UI settings
- File upload limits

## Features Explained

### Resume Parsing
The application uses LLMs to intelligently extract:
- Technical skills and technologies
- Project descriptions and technologies used
- Work experience and achievements
- Duration of experiences

### Job Matching Algorithm
The matching system considers:
- Required vs desired skills alignment
- Experience level compatibility
- Relevant project experience
- Overall profile fit

### Email Generation
Generated emails include:
- Professional subject line
- Personalized introduction
- Relevant project highlights
- Experience mentions
- Clear call-to-action
- Appropriate closing

## Troubleshooting

### Common Issues

**"API_KEY environment variable is required"**
- Make sure you created a `.env` file with your Groq API key
- Verify the key is correct and active

**"No jobs found in the content"**
- Try a different careers page URL
- Ensure the URL contains actual job listings
- Some career pages may be JavaScript-heavy and not scrapable

**"Could not parse resume properly"**
- Ensure your resume is a valid PDF
- Check that the resume has readable text (not just images)
- Try a resume with clear sections for skills, projects, and experience

**Import errors with `app` module**
- Make sure you're running from the `genAI` directory
- Try: `python -m streamlit run app/main.py`

## Technology Stack

- **Streamlit**: Web UI framework
- **LangChain**: LLM orchestration and chains
- **Groq**: Fast LLM inference
- **PyPDF2**: PDF text extraction
- **BeautifulSoup**: Web scraping
- **Python-dotenv**: Environment management

## Recent Enhancements (v2.0.0)

✅ Completed:
- [x] Support for DOCX resumes
- [x] Advanced skill matching with fuzzy matching algorithms
- [x] Detailed scoring system for job matches
- [x] Visual score breakdown in UI
- [x] Docker support for easy deployment
- [x] Comprehensive logging system
- [x] Type hints and docstrings throughout codebase
- [x] Centralized configuration management
- [x] Multiple email tone options

## Future Enhancements

Planned features for future versions:
- [ ] Multiple resume management
- [ ] Job application tracking
- [ ] Email history and templates
- [ ] Company research integration using web search
- [ ] Async processing for faster results
- [ ] Database for user sessions (PostgreSQL/SQLite)
- [ ] Browser extension for one-click generation
- [ ] Resume optimization suggestions
- [ ] LinkedIn integration

## Contributing

This is a personal project, but suggestions and feedback are welcome!

## License

MIT License - feel free to use and modify for your own purposes.

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Made for students | Powered by LangChain + Groq + Streamlit**
