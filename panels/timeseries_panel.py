import streamlit as st, pandas as pd
from loaders.mmf_loader import load_latest_ici_mmf_flows

def render_base_charts(m1, m2, y10, spx, vix):
    st.subheader("Time Series")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**M1 & M2 (Monthly, last value)**")
        m1m = m1.resample("ME").last()
        m2m = m2.resample("ME").last()
        st.line_chart(pd.DataFrame({"M1": m1m, "M2": m2m}))
    with colB:
        st.markdown("**10-Year Treasury Yield (%)**")
        st.line_chart(y10)

    colC, colD = st.columns(2)
    with colC:
        st.markdown("**S&P 500 (Close)**")
        st.line_chart(spx)
    with colD:
        st.markdown("**VIX (Close)**")
        st.line_chart(vix)

def render_mmf_section(auto_mmf: bool, mmf_file):
    st.markdown("---")
    st.subheader("Money Market Fund (MMF) Flows")

    mmf_df, mmf_msg = None, ""
    try:
        if auto_mmf:
            mmf_df = load_latest_ici_mmf_flows()
            mmf_msg = "Auto-downloaded latest ICI MMF workbook."
        elif mmf_file is not None:
            import pandas as pd
            if mmf_file.name.lower().endswith(".csv"):
                mmf_df = pd.read_csv(mmf_file)
            else:
                mmf_df = pd.read_excel(mmf_file)
            mmf_msg = f"Loaded from upload: {mmf_file.name}"
    except Exception as e:
        st.warning(f"MMF auto-download/parse failed: {e}. You can upload the file manually above.")

    if mmf_df is not None:
        import pandas as pd
        if "Date" not in mmf_df.columns:
            mmf_df.columns = [str(c).strip() for c in mmf_df.columns]
            date_col = next((c for c in mmf_df.columns if "date" in c.lower()), mmf_df.columns[0])
            mmf_df[date_col] = pd.to_datetime(mmf_df[date_col], errors="coerce")
            mmf_df = mmf_df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
            mmf_df.index.name = "Date"

        st.caption(mmf_msg)

        num_cols = [c for c in mmf_df.columns if pd.api.types.is_numeric_dtype(mmf_df[c])]
        if not num_cols:
            st.warning("No numeric columns found to chart; showing table.")
            st.dataframe(mmf_df.tail(50), width="stretch")
        else:
            st.write("Detected numeric columns:", ", ".join(num_cols[:3]))
            st.line_chart(mmf_df[num_cols[:2]].tail(260))  # ~5 years weekly
    else:
        st.info("Enable **Auto-download ICI MMF flows** or upload a file to see the chart.")
