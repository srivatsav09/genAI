"""
Script to fetch H1B sponsorship data from public sources and populate CompanyRAG.

Data source: H1B Salary Database (h1bdata.info) - Public H1B disclosure data
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from bs4 import BeautifulSoup
import time
import json
from typing import List, Dict, Any
from collections import defaultdict

from app.company_rag import CompanyRAG
from app.logger import setup_logger

logger = setup_logger(__name__)


def fetch_top_h1b_sponsors(year: int = 2024, limit: int = 200) -> List[Dict[str, Any]]:
    """
    Fetch top H1B sponsoring companies from h1bdata.info.

    Args:
        year: Year to fetch data for
        limit: Number of top companies to fetch

    Returns:
        List of company dictionaries with H1B data
    """
    logger.info(f"Fetching top {limit} H1B sponsors for year {year}")

    # h1bdata.info URL for top employers
    base_url = "https://h1bdata.info/index.php"

    companies = []

    try:
        # Fetch the page
        params = {
            "em": "",
            "job": "",
            "city": "",
            "year": str(year)
        }

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        logger.info(f"Fetching data from h1bdata.info for year {year}")
        response = requests.get(base_url, params=params, headers=headers, timeout=30)
        response.raise_for_status()

        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find table with company data
        tables = soup.find_all('table')

        if not tables:
            logger.warning("No tables found on page")
            return []

        # Find the employers table (usually the main data table)
        employer_table = None
        for table in tables:
            if table.find('th', string=lambda t: t and 'Employer' in t):
                employer_table = table
                break

        if not employer_table:
            logger.warning("Could not find employer table")
            return []

        # Parse table rows
        rows = employer_table.find_all('tr')[1:]  # Skip header

        for row in rows[:limit]:
            cols = row.find_all('td')

            if len(cols) >= 3:
                company_name = cols[0].get_text(strip=True)
                filings = cols[1].get_text(strip=True).replace(',', '')
                avg_salary = cols[2].get_text(strip=True).replace('$', '').replace(',', '')

                try:
                    company_data = {
                        'company_name': company_name,
                        'h1b_filings': int(filings) if filings.isdigit() else 0,
                        'avg_salary': float(avg_salary) if avg_salary.replace('.', '').isdigit() else 0,
                        'years_active': [year]
                    }
                    companies.append(company_data)
                    logger.info(f"Parsed: {company_name} ({filings} filings, ${avg_salary} avg)")

                except (ValueError, AttributeError) as e:
                    logger.warning(f"Error parsing row: {e}")
                    continue

        logger.info(f"Successfully fetched {len(companies)} companies")
        return companies

    except requests.RequestException as e:
        logger.error(f"Error fetching H1B data: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return []


def enrich_company_data(companies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enrich company data with common job roles and locations.

    Args:
        companies: List of company dictionaries

    Returns:
        Enriched company list
    """
    logger.info(f"Enriching data for {len(companies)} companies")

    # Common H1B roles by company type (heuristic-based)
    tech_roles = [
        "Software Engineer", "Software Developer", "Data Scientist",
        "Machine Learning Engineer", "DevOps Engineer", "Full Stack Developer",
        "Backend Engineer", "Frontend Engineer", "Data Engineer"
    ]

    consulting_roles = [
        "Business Analyst", "Consultant", "Project Manager",
        "Systems Analyst", "Technical Consultant"
    ]

    # Common H1B locations
    tech_hubs = ["San Francisco, CA", "Seattle, WA", "New York, NY", "Austin, TX", "Boston, MA"]
    consulting_hubs = ["New York, NY", "Chicago, IL", "Atlanta, GA", "Dallas, TX"]

    for company in companies:
        company_name_lower = company['company_name'].lower()

        # Heuristic: Assign roles based on company name
        if any(tech in company_name_lower for tech in ['tech', 'software', 'systems', 'google', 'amazon', 'microsoft', 'meta', 'apple', 'netflix']):
            company['roles_offered'] = tech_roles
            company['locations'] = tech_hubs
        elif any(consulting in company_name_lower for consulting in ['consulting', 'accenture', 'deloitte', 'cognizant', 'infosys', 'tcs']):
            company['roles_offered'] = consulting_roles
            company['locations'] = consulting_hubs
        else:
            # Default to mix
            company['roles_offered'] = tech_roles[:5]
            company['locations'] = tech_hubs[:3]

    logger.info("Enrichment complete")
    return companies


def save_to_json(companies: List[Dict[str, Any]], filename: str = "h1b_companies.json"):
    """Save companies to JSON file."""
    filepath = Path(__file__).parent / filename

    with open(filepath, 'w') as f:
        json.dump(companies, f, indent=2)

    logger.info(f"Saved {len(companies)} companies to {filepath}")


def load_companies_to_rag(companies: List[Dict[str, Any]]) -> int:
    """
    Load companies into CompanyRAG system.

    Args:
        companies: List of company dictionaries

    Returns:
        Number of companies loaded
    """
    logger.info("Initializing CompanyRAG")

    try:
        company_rag = CompanyRAG()

        logger.info(f"Loading {len(companies)} companies into RAG system")
        count = company_rag.store_companies_bulk(companies)

        # Get stats
        stats = company_rag.get_company_stats()
        logger.info(f"CompanyRAG stats: {stats}")

        return count

    except Exception as e:
        logger.error(f"Error loading companies to RAG: {e}")
        return 0


def main():
    """Main function to fetch and load H1B data."""
    logger.info("=" * 60)
    logger.info("H1B Data Fetcher")
    logger.info("=" * 60)

    # Configuration
    YEAR = 2024
    LIMIT = 100  # Top 100 H1B sponsors

    # Step 1: Fetch data
    logger.info(f"\nStep 1: Fetching top {LIMIT} H1B sponsors for {YEAR}")
    companies = fetch_top_h1b_sponsors(year=YEAR, limit=LIMIT)

    if not companies:
        logger.error("No companies fetched. Exiting.")
        return

    logger.info(f"✓ Fetched {len(companies)} companies")

    # Step 2: Enrich data
    logger.info("\nStep 2: Enriching company data with roles and locations")
    companies = enrich_company_data(companies)
    logger.info("✓ Data enrichment complete")

    # Step 3: Save to JSON
    logger.info("\nStep 3: Saving to JSON file")
    save_to_json(companies)
    logger.info("✓ Saved to JSON")

    # Step 4: Load to RAG
    logger.info("\nStep 4: Loading companies into CompanyRAG system")
    count = load_companies_to_rag(companies)

    if count > 0:
        logger.info(f"✓ Successfully loaded {count} companies into RAG")
    else:
        logger.error("✗ Failed to load companies into RAG")

    logger.info("\n" + "=" * 60)
    logger.info("Data fetch complete!")
    logger.info(f"Total companies in database: {count}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
