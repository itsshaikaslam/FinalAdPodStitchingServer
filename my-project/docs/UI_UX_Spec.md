UI_UX_Spec.md - Frontend Behavior Definitions
Overview
This document defines the UI/UX specifications for the Streamlit frontend of the Ad-Pod Stitching Server Optimization application.
Page Layout

Config: page_title="Ad Server Optimizer", layout="wide".
Sections:

Header: Title, problem description.
Upload: Two file_uploaders (csv only).
Action: "Optimize" button.
Results: Conditional on success: Subheaders for activated servers, assignments table, cost metrics.
Visuals: Scatter plot (locations colored by assignment), bar chart (costs).



UX Behaviors

Loading: st.spinner during API call.
Errors: st.error for 400/500 responses.
Interactivity: Download button for JSON/CSV results.
Accessibility: Use alt text for plots, clear labels.
Responsive: Wide layout for tables/plots.

Visual Style

Colors: Streamlit defaults, custom for assignments (e.g., hue by server).
Plots: Altair for interactivity.

This spec ensures intuitive, user-friendly interactions.