# Economic Complexity in Sub-Saharan Africa: Power BI Analysis

**Team Project | Power BI, R, Data Cleaning & Visualization | Final Grade: 97%**

## Overview

Analyzed the **Economic Complexity Index (ECI)** for 30 Sub-Saharan African countries (1995–2018) using a panel dataset of 500 country-year observations. As part of a five-person team, we developed an interactive Power BI dashboard to explore regional trends, changes over time, and relationships between ECI and economic, environmental, and tourism indicators.

## My Contribution: Data Preparation

Led data cleaning and preprocessing to ensure reliable analysis by:

* Designing a **hierarchical missing-value imputation** strategy (country → year → global median, or vice versa depending on the variable).
* Using **median imputation** for skewed economic indicators such as GDP and CO₂ emissions to reduce the impact of outliers.
* Treating missing **coal rents** as true zeros, reflecting countries without coal sectors rather than missing data.

This produced a clean, consistent dataset that supported the team's visualizations and analysis.

## Key Findings

* Southern Africa had the highest average ECI, while Central Africa consistently recorded the lowest.
* Regional ECI improved gradually over time, although countries such as **South Africa** declined while **Tanzania**, **Angola**, and **Uganda** showed steady gains.
* Tourism arrivals and CO₂ emissions had the strongest positive correlation with ECI, while coal rents and energy inefficiency were negatively associated.
* Results suggest that economic diversification, trade infrastructure, and industrial development are more strongly linked to economic complexity than environmental performance alone.

## Tools & Techniques

Power BI, R, panel data analysis, hierarchical data imputation, choropleth and bubble maps, scatterplots with trendlines, interactive dashboards, and correlation analysis.

Presented as a 20-minute interactive walkthrough.
