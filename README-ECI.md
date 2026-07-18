# Economic Complexity in Sub-Saharan Africa: A Power BI Analysis

**Team project | Power BI, R, Data Cleaning & Visualization**

## Overview

This project analyzed the **Economic Complexity Index (ECI)**, a measure of how sophisticated and diversified a country's productive economy is, for 30 sub-Saharan African countries over a 24-year period (1995 to 2018), using a panel dataset of 500 country-year observations.

Working in a team of five, we built an interactive Power BI report to answer four questions: how ECI is distributed across countries in the region, how it has changed over time, which economic and environmental variables are related to it, and whether richer, cleaner, or more visited countries tend to have higher ECI.

## My Contribution: Data Preparation & Cleaning

I led the data preparation stage, which underpinned the reliability of the entire analysis. Key decisions I made:

**Hierarchical missing-value imputation.** Rather than a single blanket approach, I applied a country then year then global median fallback (or year then country then global, depending on the variable), matched to each variable's underlying behavior. For example, tourism and air transport used year-level imputation first, since these are more affected by global shocks such as financial crises, while GDP and CO2 used country-level imputation first, since these reflect structural, country-specific patterns.

**Median over mean.** This was chosen specifically because economic indicators like GDP and emissions are heavily skewed, and the median is far more robust to outliers than the mean.

**Coal rents as true zero.** Rather than imputing, I recognized that missing coal rent values reflected countries with no coal sector at all, so these were set to zero rather than estimated.

This groundwork meant the rest of the team could build visuals on a clean, structurally sound dataset of 500 observations across 30 countries, without the distortion that naive imputation would have introduced.

## Key Findings

**Distribution across the region.** Southern Africa shows the highest average ECI, reflecting a historical concentration of industrial and service-based economies. Central Africa consistently shows the lowest ECI, with most countries losing complexity over time. East Africa is more mixed, with some standout improvers, and West Africa shows wide variation with no clear regional pattern.

**Change over time.** The region has seen slow, modest improvement in economic complexity over two decades. But this masks a divide: former regional leaders like **South Africa** have actually lost complexity over time, while smaller economies like **Tanzania** have shown consistent, steady improvement.

**What's driving the improvers?** Angola and Uganda stand out, still low on absolute ECI, but improving faster than most neighbors. Three factors explain this. First, economic diversification: Angola is channeling oil revenue into agriculture, telecoms, and mining, while Uganda is shifting from raw agricultural exports toward processed goods and consumer products. Second, trade and transport infrastructure: Angola has invested in routes linking mining regions to export ports, and Uganda benefits from stronger regional trade access through the African Continental Free Trade Area (AfCFTA). Third, long-term development planning: Uganda's "Tenfold Growth Strategy" explicitly targets reduced dependence on single export sectors in favor of industrial growth.

Meanwhile, South Africa's declining ECI doesn't erase its role as the region's industrial anchor; it still leads in manufacturing capacity, technology, and technical skills. Through regional trade partnerships, neighbors like Botswana, Lesotho, and Eswatini have an opening to grow agro-processing and light manufacturing, but only if they also invest in infrastructure, education, and technical skills themselves.

**What's correlated with ECI?** Tourism arrivals and CO2 emissions per capita show the strongest positive relationship: more visited, more industrially active countries tend to be more complex. GDP per capita and air transport show a moderate positive relationship, though it isn't linear. Energy inefficiency and coal rents show a negative relationship: countries reliant on coal or energy-intensive processes tend to score lower, consistent with a resource curse pattern.

Interestingly, cleaner countries did not have higher ECI; the opposite was observed, since industrial activity drives both complexity and emissions together. More visited countries did show higher ECI, making tourism one of the strongest positive correlates in the dataset.

## Tools & Techniques

Power BI (interactive report with slicers and filters across year and country category), R script visual for statistical analysis, hierarchical data imputation, panel data structuring, choropleth and bubble mapping, correlation analysis via scatterplots with trendlines.

Team project, presented as a 20-minute interactive walkthrough to the class.
