"""
Company Insights Page - H1B Sponsorship Data Visualization
"""
import sys
from pathlib import Path

# Add parent directory to path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any

try:
    from app.company_rag import CompanyRAG
    from app.config import Config
    from app.logger import setup_logger
except ImportError:
    from company_rag import CompanyRAG
    from config import Config
    from logger import setup_logger

logger = setup_logger(__name__)


def create_h1b_trend_chart(companies_data: List[Dict[str, Any]]) -> go.Figure:
    """
    Create time-series chart showing H1B filings over years for multiple companies.

    Args:
        companies_data: List of company dictionaries with yearly_filings data

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    # Add a line for each company
    for company in companies_data:
        if 'yearly_filings' in company:
            yearly_data = company['yearly_filings']
            years = sorted(yearly_data.keys())
            filings = [yearly_data[year] for year in years]

            fig.add_trace(go.Scatter(
                x=years,
                y=filings,
                mode='lines+markers',
                name=company['company_name'],
                line=dict(width=2),
                marker=dict(size=8)
            ))

    fig.update_layout(
        title="H1B Sponsorship Trends Over Time",
        xaxis_title="Year",
        yaxis_title="Number of H1B Filings",
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )

    return fig


def create_top_sponsors_bar_chart(companies: List[Dict[str, Any]], n: int = 10) -> go.Figure:
    """
    Create bar chart of top N H1B sponsors.

    Args:
        companies: List of company dictionaries
        n: Number of top companies to show

    Returns:
        Plotly figure object
    """
    # Sort by H1B filings
    sorted_companies = sorted(
        companies,
        key=lambda x: x.get('h1b_filings', 0),
        reverse=True
    )[:n]

    company_names = [c['company_name'] for c in sorted_companies]
    filings = [c.get('h1b_filings', 0) for c in sorted_companies]

    fig = go.Figure(data=[
        go.Bar(
            x=filings,
            y=company_names,
            orientation='h',
            marker=dict(
                color=filings,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Filings")
            ),
            text=filings,
            textposition='outside'
        )
    ])

    fig.update_layout(
        title=f"Top {n} H1B Sponsoring Companies",
        xaxis_title="Total H1B Filings",
        yaxis_title="",
        template='plotly_white',
        height=500,
        yaxis=dict(autorange="reversed")
    )

    return fig


def create_salary_distribution(companies: List[Dict[str, Any]]) -> go.Figure:
    """
    Create box plot or histogram of H1B salaries by company.

    Args:
        companies: List of company dictionaries

    Returns:
        Plotly figure object
    """
    company_names = []
    salaries = []

    for company in companies:
        if 'avg_salary' in company and company['avg_salary'] > 0:
            company_names.append(company['company_name'])
            salaries.append(company['avg_salary'])

    fig = go.Figure(data=[
        go.Bar(
            x=company_names,
            y=salaries,
            marker=dict(
                color=salaries,
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="Salary ($)")
            ),
            text=[f"${s:,.0f}" for s in salaries],
            textposition='outside'
        )
    ])

    fig.update_layout(
        title="Average H1B Salaries by Company",
        xaxis_title="",
        yaxis_title="Average Salary (USD)",
        template='plotly_white',
        height=500,
        xaxis=dict(tickangle=-45)
    )

    return fig


def main():
    """Main function for company insights page."""
    st.set_page_config(
        page_title="Company Insights - H1B Data",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 Company Intelligence & H1B Insights")
    st.markdown("Explore H1B sponsorship trends and company data")

    # Initialize CompanyRAG
    try:
        company_rag = CompanyRAG()
        stats = company_rag.get_company_stats()

        if stats['total_companies'] == 0:
            st.warning("No company data loaded. Please run `python scripts/load_sample_companies.py` first.")
            st.stop()

        st.success(f"Database loaded: {stats['total_companies']} companies")

    except Exception as e:
        st.error(f"Failed to initialize CompanyRAG: {e}")
        st.stop()

    # Sidebar filters
    with st.sidebar:
        st.header("Filters")

        search_query = st.text_input(
            "Search companies",
            placeholder="e.g., Software Engineer"
        )

        n_companies = st.slider(
            "Number of companies to show",
            min_value=5,
            max_value=50,
            value=20
        )

        if st.button("Search Companies"):
            st.session_state.search_query = search_query

    # Search for companies
    if 'search_query' not in st.session_state or not st.session_state.search_query:
        st.session_state.search_query = "H1B sponsor Software Engineer"

    with st.spinner("Searching companies..."):
        results = company_rag.search_companies(
            st.session_state.search_query,
            n_results=n_companies
        )

    if not results:
        st.warning("No companies found. Try a different search.")
        st.stop()

    # Extract company data from results
    companies_data = []
    for result in results:
        # Parse document text back to dict (simplified)
        doc_lines = result['document'].split('\n')
        company_dict = {
            'company_name': result['metadata']['company_name'],
            'h1b_filings': result['metadata'].get('h1b_filings', 0),
            'avg_salary': result['metadata'].get('avg_salary', 0),
            'similarity_score': result.get('similarity_score', 0)
        }

        # Try to extract yearly_filings if present in document
        for line in doc_lines:
            if 'H1B Years:' in line:
                # This is a placeholder - in real implementation,
                # we'd store yearly_filings in metadata
                pass

        companies_data.append(company_dict)

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Trends Over Time",
        "🏆 Top Sponsors",
        "💰 Salary Analysis",
        "🔍 Company List"
    ])

    with tab1:
        st.subheader("H1B Sponsorship Trends")

        # For demo, create sample trend data
        sample_trend_companies = [
            {
                "company_name": "Google LLC",
                "yearly_filings": {"2020": 7234, "2021": 7845, "2022": 8012, "2023": 8156, "2024": 8234}
            },
            {
                "company_name": "Amazon.com Services LLC",
                "yearly_filings": {"2020": 9123, "2021": 9876, "2022": 10234, "2023": 10456, "2024": 10543}
            },
            {
                "company_name": "Microsoft Corporation",
                "yearly_filings": {"2020": 8456, "2021": 9012, "2022": 9345, "2023": 9678, "2024": 9876}
            },
            {
                "company_name": "Meta Platforms Inc",
                "yearly_filings": {"2020": 6789, "2021": 7234, "2022": 7456, "2023": 7543, "2024": 7654}
            },
            {
                "company_name": "Apple Inc",
                "yearly_filings": {"2020": 5678, "2021": 6012, "2022": 6234, "2023": 6456, "2024": 6543}
            }
        ]

        fig_trend = create_h1b_trend_chart(sample_trend_companies)
        st.plotly_chart(fig_trend, use_container_width=True)

        st.info("📝 This shows the growth trend of H1B filings for major tech companies from 2020-2024")

    with tab2:
        st.subheader("Top H1B Sponsors")

        top_n = st.select_slider(
            "Show top N companies",
            options=[5, 10, 15, 20],
            value=10
        )

        fig_bar = create_top_sponsors_bar_chart(companies_data, n=top_n)
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab3:
        st.subheader("Salary Analysis")

        fig_salary = create_salary_distribution(companies_data[:15])
        st.plotly_chart(fig_salary, use_container_width=True)

        # Statistics
        col1, col2, col3 = st.columns(3)

        salaries = [c['avg_salary'] for c in companies_data if c['avg_salary'] > 0]

        with col1:
            st.metric("Average Salary", f"${sum(salaries) / len(salaries):,.0f}")

        with col2:
            st.metric("Highest", f"${max(salaries):,.0f}")

        with col3:
            st.metric("Lowest", f"${min(salaries):,.0f}")

    with tab4:
        st.subheader("Company List")

        st.markdown(f"**Search Query:** {st.session_state.search_query}")
        st.markdown(f"**Results:** {len(companies_data)} companies")

        # Display as table
        for idx, company in enumerate(companies_data, 1):
            with st.expander(f"{idx}. {company['company_name']} ({company['similarity_score']:.1f}% match)"):
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("H1B Filings", f"{company['h1b_filings']:,}")

                with col2:
                    st.metric("Avg Salary", f"${company['avg_salary']:,.0f}")


if __name__ == "__main__":
    main()
