"""UI theme and CSS injection for Field Shift."""
import streamlit as st


def apply_custom_theme(is_urdu: bool = False) -> None:
    direction = "rtl" if is_urdu else "ltr"
    text_align = "right" if is_urdu else "left"

    st.markdown(
        f"""
        <style>
        html, body, [class*="css"] {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #111827;
            direction: {direction};
            text-align: {text_align};
        }}
        .top-navbar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #E5E7EB;
            padding-bottom: 12px;
            margin-bottom: 24px;
        }}
        .brand-title {{
            font-size: 1.4rem;
            font-weight: 800;
            color: #1B4D3E;
            letter-spacing: -0.02em;
        }}
        .brand-subtitle {{
            font-size: 0.82rem;
            color: #6B7280;
            font-weight: 500;
        }}
        .status-pill {{
            display: inline-block;
            background-color: #ECFDF5;
            color: #065F46;
            border: 1px solid #A7F3D0;
            border-radius: 9999px;
            padding: 4px 12px;
            font-size: 0.78rem;
            font-weight: 600;
        }}
        .app-card {{
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 16px;
        }}
        .app-card-featured {{
            border: 1.5px solid #059669;
            background: #FBFDFB;
        }}
        .card-header-label {{
            font-size: 0.72rem;
            font-weight: 700;
            color: #6B7280;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 4px;
        }}
        .card-title-lg {{
            font-size: 1.15rem;
            font-weight: 700;
            color: #111827;
            margin-bottom: 2px;
        }}
        .advisory-notice {{
            background-color: #FFFBEB;
            border-left: 3px solid #D97706;
            padding: 10px 14px;
            font-size: 0.88rem;
            color: #78350F;
            border-radius: 2px;
            margin-top: 14px;
        }}
        .insight-tile {{
            background: #F9FAFB;
            border: 1px solid #E5E7EB;
            border-radius: 8px;
            padding: 16px;
        }}
        .insight-label {{
            font-size: 0.72rem;
            font-weight: 700;
            color: #6B7280;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .insight-value {{
            font-size: 1.5rem;
            font-weight: 800;
            color: #111827;
            margin: 2px 0;
        }}
        .insight-desc {{
            font-size: 0.8rem;
            color: #6B7280;
        }}
        div[data-testid="stMetricValue"] {{
            font-size: 1.1rem;
            font-weight: 700;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

