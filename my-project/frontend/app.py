from __future__ import annotations

import io
import json
from typing import Dict, List

import pandas as pd
import requests
import streamlit as st


BACKEND_URL = "http://localhost:8000"


st.set_page_config(page_title="Ad Server Optimizer", layout="wide")
st.title("Ad-Pod Stitching Server Optimizer")
st.write("Upload servers.csv and dmas.csv to optimize server activation and assignments.")

servers_file = st.file_uploader("Upload servers.csv", type=["csv"])
dmas_file = st.file_uploader("Upload dmas.csv", type=["csv"])
timeout_sec = st.number_input("Request timeout (seconds)", min_value=10, max_value=300, value=120)
fast_mode = st.checkbox("Use fast greedy optimizer", value=True)

if st.button("Optimize"):
    if not servers_file or not dmas_file:
        st.error("Please upload both servers.csv and dmas.csv")
    else:
        with st.spinner("Optimizing..."):
            files = {
                "servers": ("servers.csv", servers_file.getvalue(), "text/csv"),
                "dmas": ("dmas.csv", dmas_file.getvalue(), "text/csv"),
            }
            try:
                endpoint = "/optimize_fast" if fast_mode else "/optimize"
                resp = requests.post(f"{BACKEND_URL}{endpoint}", files=files, timeout=float(timeout_sec))
                if resp.status_code != 200:
                    st.error(f"Error {resp.status_code}: {resp.text}")
                else:
                    data = resp.json()
                    st.subheader("Activated Servers")
                    st.write(data.get("activated_servers", []))

                    st.subheader("Assignments")
                    assignments = data.get("assignments", {})
                    if assignments:
                        df = pd.DataFrame(list(assignments.items()), columns=["DMA ID", "Server ID"])
                        st.dataframe(df, use_container_width=True)

                    st.subheader("Costs")
                    costs = data.get("costs", {})
                    cols = st.columns(3)
                    cols[0].metric("Total Setup Cost", f"{costs.get('total_setup_cost', 0):.2f}")
                    cols[1].metric("Total Delivery Cost", f"{costs.get('total_delivery_cost', 0):.2f}")
                    cols[2].metric("Total Cost", f"{costs.get('total_cost', 0):.2f}")

                    st.download_button(
                        "Download JSON", data=json.dumps(data, indent=2), file_name="optimization_result.json"
                    )
            except requests.RequestException as exc:
                st.error(f"Request failed: {exc}")


