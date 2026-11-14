"""Reusable UI components for Field Shift."""
import streamlit as st


def top_header(is_urdu: bool = False) -> None:
    title = "فیلڈ شفٹ" if is_urdu else "Field Shift"
    subtitle = "فصل کی گردش کا منصوبہ ساز | ناسا موسمیاتی ڈیٹا" if is_urdu else "Crop Rotation Decision Support | NASA Agro-Climatology"
    badge = "ناسا اسپیس ایپس 2026" if is_urdu else "NASA Space Apps 2026"
    st.markdown(
        f"""
        <div class="top-navbar">
            <div>
                <div class="brand-title">{title}</div>
                <div class="brand-subtitle">{subtitle}</div>
            </div>
            <span class="status-pill">{badge}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def farm_profile_card(site_name: str, soil: str, water: str, rotation_name: str, status: str = "Ready") -> None:
    st.markdown(
        f"""
        <div class="app-card" style="margin-bottom: 20px;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #6B7280; text-transform: uppercase; margin-bottom: 6px;">Current Farm Profile</div>
            <div style="font-size: 1.0rem; font-weight: 700; color: #111827; margin-bottom: 10px;">{site_name}</div>
            <div style="display: flex; gap: 24px; flex-wrap: wrap;">
                <div><span style="font-size: 0.75rem; color: #6B7280; display: block;">Soil Type</span><span style="font-weight: 600; font-size: 0.9rem;">{soil}</span></div>
                <div><span style="font-size: 0.75rem; color: #6B7280; display: block;">Water Source</span><span style="font-weight: 600; font-size: 0.9rem;">{water}</span></div>
                <div><span style="font-size: 0.75rem; color: #6B7280; display: block;">Current Rotation</span><span style="font-weight: 600; font-size: 0.9rem;">{rotation_name}</span></div>
                <div><span style="font-size: 0.75rem; color: #6B7280; display: block;">Status</span><span class="status-pill">{status}</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight_card(col, label: str, value: str, description: str, source: str = "") -> None:
    col.markdown(
        f"""
        <div class="insight-tile">
            <div class="insight-label">{label}</div>
            <div class="insight-value">{value}</div>
            <div class="insight-desc">{description}</div>
            {f'<div style="font-size: 0.7rem; color: #9CA3AF; margin-top: 4px;">{source}</div>' if source else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def persistent_expert_banner(is_urdu: bool = False) -> None:
    msg = (
        "یہ آلہ فیصلہ سازی کی مدد کرتا ہے، زرعی ماہر کی جگہ نہیں لیتا۔ مقامی زرعی توسیع افسر سے مشورہ کریں۔"
        if is_urdu
        else "This tool provides data-informed guidance to support — not replace — advice from a qualified local agronomist or OFWM extension officer."
    )
    st.markdown(
        f"""
        <div style="margin-top: 32px; padding: 12px 16px; background: #F3F4F6; border-radius: 6px;
                    font-size: 0.82rem; color: #374151; border-left: 3px solid #9CA3AF;">
            {msg}
        </div>
        """,
        unsafe_allow_html=True,
    )

