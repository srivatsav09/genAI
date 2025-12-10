"""
Load sample H1B sponsoring companies into CompanyRAG for testing.
This uses realistic data based on known H1B sponsors.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.company_rag import CompanyRAG
from app.logger import setup_logger

logger = setup_logger(__name__)


# Sample companies based on real H1B sponsors
SAMPLE_COMPANIES = [
    {
        "company_name": "Google LLC",
        "h1b_filings": 8234,
        "avg_salary": 145000,
        "roles_offered": ["Software Engineer", "Data Scientist", "ML Engineer", "Product Manager", "SRE"],
        "locations": ["Mountain View, CA", "New York, NY", "Seattle, WA", "Austin, TX"],
        "years_active": [2020, 2021, 2022, 2023, 2024],
        "yearly_filings": {"2020": 7234, "2021": 7845, "2022": 8012, "2023": 8156, "2024": 8234}
    },
    {
        "company_name": "Amazon.com Services LLC",
        "h1b_filings": 10543,
        "avg_salary": 135000,
        "roles_offered": ["Software Development Engineer", "Data Engineer", "ML Scientist", "DevOps Engineer"],
        "locations": ["Seattle, WA", "Arlington, VA", "Austin, TX", "San Francisco, CA"],
        "years_active": [2020, 2021, 2022, 2023, 2024]
    },
    {
        "company_name": "Microsoft Corporation",
        "h1b_filings": 9876,
        "avg_salary": 140000,
        "roles_offered": ["Software Engineer", "Cloud Engineer", "Data Scientist", "Security Engineer"],
        "locations": ["Redmond, WA", "San Francisco, CA", "New York, NY"],
        "years_active": [2020, 2021, 2022, 2023, 2024]
    },
    {
        "company_name": "Meta Platforms Inc",
        "h1b_filings": 7654,
        "avg_salary": 155000,
        "roles_offered": ["Software Engineer", "Research Scientist", "Data Engineer", "ML Engineer"],
        "locations": ["Menlo Park, CA", "Seattle, WA", "New York, NY"],
        "years_active": [2020, 2021, 2022, 2023, 2024]
    },
    {
        "company_name": "Apple Inc",
        "h1b_filings": 6543,
        "avg_salary": 150000,
        "roles_offered": ["Software Engineer", "Hardware Engineer", "ML Engineer", "iOS Developer"],
        "locations": ["Cupertino, CA", "Austin, TX", "Seattle, WA"],
        "years_active": [2020, 2021, 2022, 2023, 2024]
    },
    {
        "company_name": "Cognizant Technology Solutions",
        "h1b_filings": 15432,
        "avg_salary": 85000,
        "roles_offered": ["Software Developer", "Business Analyst", "Systems Analyst", "Consultant"],
        "locations": ["Teaneck, NJ", "Irving, TX", "San Francisco, CA", "Phoenix, AZ"],
        "years_active": [2020, 2021, 2022, 2023, 2024]
    },
    {
        "company_name": "Infosys Limited",
        "h1b_filings": 14321,
        "avg_salary": 82000,
        "roles_offered": ["Technology Analyst", "Software Engineer", "Consultant", "Systems Engineer"],
        "locations": ["Richardson, TX", "Hartford, CT", "Atlanta, GA"],
        "years_active": [2020, 2021, 2022, 2023, 2024]
    },
    {
        "company_name": "Tata Consultancy Services",
        "h1b_filings": 13210,
        "avg_salary": 80000,
        "roles_offered": ["Software Developer", "IT Analyst", "Systems Engineer", "Consultant"],
        "locations": ["New York, NY", "Cincinnati, OH", "Atlanta, GA"],
        "years_active": [2020, 2021, 2022, 2023, 2024]
    },
    {
        "company_name": "Accenture LLP",
        "h1b_filings": 8765,
        "avg_salary": 95000,
        "roles_offered": ["Software Engineer", "Consultant", "Business Analyst", "Cloud Architect"],
        "locations": ["New York, NY", "Chicago, IL", "San Francisco, CA"],
        "years_active": [2020, 2021, 2022, 2023, 2024]
    },
    {
        "company_name": "Deloitte Consulting LLP",
        "h1b_filings": 7890,
        "avg_salary": 98000,
        "roles_offered": ["Consultant", "Business Analyst", "Data Analyst", "Software Engineer"],
        "locations": ["New York, NY", "Chicago, IL", "San Francisco, CA", "Atlanta, GA"],
        "years_active": [2020, 2021, 2022, 2023, 2024]
    },
    {
        "company_name": "IBM Corporation",
        "h1b_filings": 6789,
        "avg_salary": 110000,
        "roles_offered": ["Software Engineer", "Data Scientist", "Cloud Engineer", "AI Researcher"],
        "locations": ["Armonk, NY", "Austin, TX", "San Francisco, CA"],
        "years_active": [2020, 2021, 2022, 2023, 2024]
    },
    {
        "company_name": "Oracle America Inc",
        "h1b_filings": 5678,
        "avg_salary": 125000,
        "roles_offered": ["Software Engineer", "Database Administrator", "Cloud Engineer", "Solutions Architect"],
        "locations": ["Redwood City, CA", "Austin, TX", "Seattle, WA"],
        "years_active": [2020, 2021, 2022, 2023, 2024]
    },
    {
        "company_name": "Intel Corporation",
        "h1b_filings": 5234,
        "avg_salary": 130000,
        "roles_offered": ["Hardware Engineer", "Software Engineer", "Validation Engineer", "ML Engineer"],
        "locations": ["Santa Clara, CA", "Hillsboro, OR", "Austin, TX"],
        "years_active": [2020, 2021, 2022, 2023, 2024]
    },
    {
        "company_name": "Qualcomm Technologies Inc",
        "h1b_filings": 4567,
        "avg_salary": 135000,
        "roles_offered": ["Software Engineer", "Hardware Engineer", "5G Engineer", "ML Engineer"],
        "locations": ["San Diego, CA", "Santa Clara, CA", "Austin, TX"],
        "years_active": [2020, 2021, 2022, 2023, 2024]
    },
    {
        "company_name": "Salesforce Inc",
        "h1b_filings": 4321,
        "avg_salary": 140000,
        "roles_offered": ["Software Engineer", "Solutions Engineer", "Data Scientist", "Product Manager"],
        "locations": ["San Francisco, CA", "Seattle, WA", "Indianapolis, IN"],
        "years_active": [2020, 2021, 2022, 2023, 2024]
    },
    {
        "company_name": "Netflix Inc",
        "h1b_filings": 1234,
        "avg_salary": 185000,
        "roles_offered": ["Software Engineer", "Data Engineer", "ML Engineer", "Content Engineer"],
        "locations": ["Los Gatos, CA", "Los Angeles, CA"],
        "years_active": [2020, 2021, 2022, 2023, 2024]
    },
    {
        "company_name": "Adobe Inc",
        "h1b_filings": 3456,
        "avg_salary": 140000,
        "roles_offered": ["Software Engineer", "ML Engineer", "Data Scientist", "UX Engineer"],
        "locations": ["San Jose, CA", "Seattle, WA", "Lehi, UT"],
        "years_active": [2020, 2021, 2022, 2023, 2024]
    },
    {
        "company_name": "Uber Technologies Inc",
        "h1b_filings": 2345,
        "avg_salary": 145000,
        "roles_offered": ["Software Engineer", "Data Scientist", "ML Engineer", "Backend Engineer"],
        "locations": ["San Francisco, CA", "Seattle, WA", "New York, NY"],
        "years_active": [2020, 2021, 2022, 2023, 2024]
    },
    {
        "company_name": "Airbnb Inc",
        "h1b_filings": 1876,
        "avg_salary": 155000,
        "roles_offered": ["Software Engineer", "Data Engineer", "ML Engineer", "Product Engineer"],
        "locations": ["San Francisco, CA", "Seattle, WA"],
        "years_active": [2020, 2021, 2022, 2023, 2024]
    },
    {
        "company_name": "LinkedIn Corporation",
        "h1b_filings": 2543,
        "avg_salary": 145000,
        "roles_offered": ["Software Engineer", "Data Scientist", "Backend Engineer", "ML Engineer"],
        "locations": ["Sunnyvale, CA", "New York, NY", "Seattle, WA"],
        "years_active": [2020, 2021, 2022, 2023, 2024]
    },
]


def main():
    """Load sample companies into CompanyRAG."""
    logger.info("=" * 60)
    logger.info("Sample Company Data Loader")
    logger.info("=" * 60)

    logger.info(f"\nLoading {len(SAMPLE_COMPANIES)} sample H1B sponsoring companies...")

    try:
        # Initialize CompanyRAG
        company_rag = CompanyRAG()

        # Load companies
        count = company_rag.store_companies_bulk(SAMPLE_COMPANIES)

        # Get stats
        stats = company_rag.get_company_stats()

        logger.info("\n" + "=" * 60)
        logger.info(f"✓ Successfully loaded {count} companies")
        logger.info(f"Database stats: {stats}")
        logger.info("=" * 60)

        # Test search
        logger.info("\nTesting search...")
        results = company_rag.search_companies("Software Engineer H1B sponsor", n_results=5)

        logger.info(f"\nTop 5 companies for 'Software Engineer H1B sponsor':")
        for idx, result in enumerate(results, 1):
            company = result['metadata']['company_name']
            score = result['similarity_score']
            logger.info(f"{idx}. {company} (similarity: {score}%)")

    except Exception as e:
        logger.error(f"Error loading companies: {e}")
        raise


if __name__ == "__main__":
    main()
