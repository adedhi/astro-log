import streamlit as st
from sqlalchemy import select
from core.database import SessionLocal, init_db
from core.models import ImagingSession
from core.rag import generate_rag_answer, hybrid_search

st.set_page_config(
    page_title="AstroLog | Astrophotography Session Log & RAG",
    page_icon="🔭",
    layout="wide",
)

init_db()

def load_sessions(target_filter: str, min_rating: int) -> list[ImagingSession]:
    """Fetches logged imaging sessions from Neon PostgreSQL with optional UI filters."""
    with SessionLocal() as db:
        stmt = select(ImagingSession).order_by(ImagingSession.captured_at.desc())
        if target_filter != "All":
            stmt = stmt.where(ImagingSession.target_name.ilike(target_filter))
        if min_rating > 1:
            stmt = stmt.where(ImagingSession.rating >= min_rating)
        return list(db.scalars(stmt).all())

def get_distinct_targets() -> list[str]:
    """Returns a sorted list of unique celestial targets logged in the database."""
    with SessionLocal() as db:
        stmt = select(ImagingSession.target_name).distinct()
        targets = [t for t in db.scalars(stmt).all() if t]
        return sorted(targets)

def render_session_card(session: ImagingSession) -> None:
    """Renders a single astrophotography session card with image, EXIF, and sky telemetry."""
    with st.container(border=True):
        col_img, col_meta = st.columns([1, 1.4], gap="large")

        with col_img:
            st.image(session.image_url, use_container_width=True)

        with col_meta:
            stars = "★" * session.rating + "☆" * (5 - session.rating)
            st.subheader(f"Session #{session.id} — {session.target_name} ({stars})")
            st.caption(
                f"Captured: **{session.captured_at.strftime('%Y-%m-%d %H:%M UTC')}** | "
                f"Optics: **{session.optics_description}**"
            )

            # Row 1: Camera EXIF
            c1, c2, c3 = st.columns(3)
            c1.metric("Camera", session.camera_model or "N/A")
            c2.metric("ISO", session.iso if session.iso is not None else "N/A")
            c3.metric("Shutter Speed", session.shutter_speed or "N/A")

            # Row 2: Astropy & Open-Meteo Telemetry
            w1, w2, w3, w4 = st.columns(4)
            alt_str = (
                f"{session.target_altitude_deg:.1f}°"
                if session.target_altitude_deg is not None
                else "N/A"
            )
            moon_str = (
                f"{session.moon_illumination_pct:.0f}%"
                if session.moon_illumination_pct is not None
                else "N/A"
            )
            wind_str = (
                f"{session.wind_speed_kmh:.1f} km/h"
                if session.wind_speed_kmh is not None
                else "N/A"
            )
            cloud_str = (
                f"{session.cloud_cover_pct}%"
                if session.cloud_cover_pct is not None
                else "N/A"
            )

            w1.metric("Target Altitude", alt_str)
            w2.metric("Moon Illum.", moon_str)
            w3.metric("Wind Speed", wind_str)
            w4.metric("Cloud Cover", cloud_str)

            if session.user_notes:
                st.markdown(f"**Observer Notes:** {session.user_notes}")

st.title("🔭 AstroLog")
st.markdown(
    "Astrophotography session archive and hybrid Retrieval-Augmented Generation (RAG) assistant. "
    "Sessions are ingested via a local CLI pipeline that extracts DSLR EXIF settings, strips personal metadata "
    "prior to Cloudflare R2 upload, computes sky ephemeris via `Astropy`, and indexes 3072-dimensional embeddings in Neon PostgreSQL (`pgvector`)."
)

tab_gallery, tab_rag = st.tabs(["Session Gallery", "Ask AstroLog (RAG)"])

# === TAB 1: SESSION GALLERY ===
with tab_gallery:
    targets = ["All"] + get_distinct_targets()
    f_col1, f_col2 = st.columns([1, 2])
    with f_col1:
        selected_target = st.selectbox("Filter by Target", options=targets)
    with f_col2:
        selected_rating = st.slider("Minimum Quality Rating", min_value=1, max_value=5, value=1)

    sessions = load_sessions(selected_target, selected_rating)
    st.caption(f"Showing **{len(sessions)}** logged imaging session(s).")

    if not sessions:
        st.info("No imaging sessions match the selected filters.")
    else:
        for s in sessions:
            render_session_card(s)

# === TAB 2: HYBRID RAG ASSISTANT ===
with tab_rag:
    st.markdown(
        "Query past imaging sessions using natural language. The engine extracts structured SQL metadata filters "
        "via Gemini and combines them with `pgvector` cosine distance search."
    )

    sample_queries = [
        "What ISO and shutter speed gave the sharpest results on the Moon?",
        "How did wind speed and target altitude affect the sessions?",
        "Show me sessions with crisp craters or minimal atmospheric boiling.",
    ]
    st.caption(f"Example queries:\n\n" + f"\n\n".join(f"• *\"{q}\"*" for q in sample_queries))

    with st.form("rag_query_form"):
        user_query = st.text_input(
            "Ask a question about logged equipment, camera settings, or sky conditions:",
            placeholder="e.g., What ISO and shutter speed worked best on the Moon?",
        )
        submitted = st.form_submit_button("Search & Synthesize")

    if submitted and user_query.strip():
        with st.spinner("Running hybrid SQL + pgvector search and synthesizing answer..."):
            with SessionLocal() as db:
                retrieved_sessions, extracted_filters = hybrid_search(
                    db=db, user_query=user_query.strip(), top_k=5
                )
                answer = generate_rag_answer(
                    user_query=user_query.strip(),
                    retrieved_sessions=retrieved_sessions,
                )

        st.markdown("### Answer")
        st.write(answer)

        with st.expander("🛠️ View Extracted Hybrid Search Filters (Pydantic -> SQL)"):
            st.json(extracted_filters.model_dump())

        if retrieved_sessions:
            st.markdown(f"### Retrieved Context ({len(retrieved_sessions)} session(s))")
            for s in retrieved_sessions:
                render_session_card(s)
