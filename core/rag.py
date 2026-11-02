from datetime import datetime
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session
from core.config import settings
from core.models import ImagingSession

client = genai.Client(api_key=settings.GEMINI_API_KEY)
RAG_SYSTEM_INSTRUCTION = """
You are an astrophotography assistant analyzing logged telescope imaging sessions.
Answer the user's question using ONLY the provided session logs below.
- Always cite specific sessions by their [Session #ID] and capture date when referencing camera settings (ISO, shutter speed), weather, or target altitude.
- Be concise, direct, and technical.
- If the retrieved sessions do not contain enough information to answer the question, state that clearly instead of guessing.
"""

class SearchFilters(BaseModel):
    target_name: str | None = Field(
        default=None,
        description="Celestial target name if explicitly mentioned (e.g., 'Moon', 'Saturn', 'Jupiter').",
    )
    min_rating: int | None = Field(
        default=None,
        description="Minimum quality rating (1-5) if user asks for 'best', 'sharpest', or a specific minimum rating.",
    )
    max_wind_speed_kmh: float | None = Field(
        default=None,
        description="Maximum wind speed in km/h if user specifies low wind or an upper wind limit.",
    )
    max_cloud_cover_pct: int | None = Field(
        default=None,
        description="Maximum cloud cover percentage (0-100) if user mentions clear skies or low clouds.",
    )

def build_document_text(
    target_name: str,
    captured_at_utc: datetime,
    optics_description: str,
    rating: int,
    camera_model: str | None,
    iso: int | None,
    shutter_speed: str | None,
    target_altitude_deg: float | None,
    target_azimuth_deg: float | None,
    moon_illumination_pct: float | None,
    cloud_cover_pct: int | None,
    wind_speed_kmh: float | None,
    temperature_c: float | None,
    user_notes: str | None,
) -> str:
    """Synthesizes session metadata and notes into a structured paragraph for vector embedding."""
    date_str = captured_at_utc.strftime("%Y-%m-%d %H:%M UTC")
    camera_str = camera_model or "Unknown Camera"
    iso_str = f"ISO {iso}" if iso is not None else "ISO Unknown"
    shutter_str = shutter_speed or "Unknown Shutter"

    sky_measurement_parts = []
    if target_altitude_deg is not None and target_azimuth_deg is not None:
        sky_measurement_parts.append(f"Target Altitude: {target_altitude_deg:.1f}°, Azimuth: {target_azimuth_deg:.1f}°")
    if moon_illumination_pct is not None:
        sky_measurement_parts.append(f"Moon Illumination: {moon_illumination_pct:.1f}%")
    if cloud_cover_pct is not None:
        sky_measurement_parts.append(f"Cloud Cover: {cloud_cover_pct}%")
    if wind_speed_kmh is not None:
        sky_measurement_parts.append(f"Wind Speed: {wind_speed_kmh:.1f} km/h")
    if temperature_c is not None:
        sky_measurement_parts.append(f"Temperature: {temperature_c:.1f}°C")

    sky_measurement_summary = " | ".join(sky_measurement_parts) if sky_measurement_parts else "No sky/weather telemetry."
    notes_str = user_notes.strip() if user_notes else "No additional observer notes."

    return (
        f"Target: {target_name} | Captured: {date_str} | Rating: {rating}/5\n"
        f"Equipment: {optics_description} | Camera: {camera_str} ({iso_str}, Shutter {shutter_str})\n"
        f"Conditions: {sky_measurement_summary}\n"
        f"Observer Notes: {notes_str}"
    )

def generate_embedding(text: str) -> list[float]:
    """Generates a 3072-dimensional vector embedding using Gemini gemini-embedding-001."""
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
    )
    return list(response.embeddings[0].values)

def extract_search_filters(user_query: str) -> SearchFilters:
    """Uses Gemini Structured Outputs to extract SQL metadata filters from a natural language query."""
    prompt = f"""
    Extract explicit search filters from the user's astrophotography query.
    - If the user asks for "low wind" or "calm wind" without a specific number, set max_wind_speed_kmh = 15.0.
    - If the user asks for "clear skies" without a number, set max_cloud_cover_pct = 25.
    - If the user asks for "best" or "sharpest" shots without a number, set min_rating = 4.
    - Leave any unmentioned filter as null.

    User Query: "{user_query}"
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SearchFilters,
            temperature=0.0,
        ),
    )
    return SearchFilters.model_validate_json(response.text)

def hybrid_search(db: Session, user_query: str, top_k: int = 5) -> tuple[list[ImagingSession], SearchFilters]:
    """
    Executes a hybrid search combining structured SQL metadata filtering with
    pgvector cosine distance ordering. Falls back to pure vector search if strict
    SQL filters match zero rows.
    """
    filters = extract_search_filters(user_query)
    query_vector = generate_embedding(user_query)
    stmt = select(ImagingSession)

    if filters.target_name:
        stmt = stmt.where(ImagingSession.target_name.ilike(f"%{filters.target_name.strip()}%"))
    if filters.min_rating is not None:
        stmt = stmt.where(ImagingSession.rating >= filters.min_rating)
    if filters.max_wind_speed_kmh is not None:
        stmt = stmt.where(ImagingSession.wind_speed_kmh <= filters.max_wind_speed_kmh)
    if filters.max_cloud_cover_pct is not None:
        stmt = stmt.where(ImagingSession.cloud_cover_pct <= filters.max_cloud_cover_pct)

    stmt = stmt.order_by(ImagingSession.embedding.cosine_distance(query_vector)).limit(top_k)
    results = list(db.scalars(stmt).all())

    # Fallback: if metadata filters were to strict and returned 0 rows, run only vector search
    if not results:
        fallback_stmt = (
            select(ImagingSession)
            .order_by(ImagingSession.embedding.cosine_distance(query_vector))
            .limit(top_k)
        )
        results = list(db.scalars(fallback_stmt).all())

    return results, filters

def generate_rag_answer(user_query: str, retrieved_sessions: list[ImagingSession]) -> str:
    """Generates a grounded, cited answer from the retrieved imaging sessions."""
    if not retrieved_sessions:
        return "No imaging sessions have been logged in the database yet."

    context_blocks = [
        f"[Session #{s.id}]\n{s.document_text}" for s in retrieved_sessions
    ]
    formatted_context = "\n\n".join(context_blocks)

    user_prompt = f"""
    === RETRIEVED IMAGING SESSIONS ===
    {formatted_context}

    === USER QUESTION ===
    {user_query}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=RAG_SYSTEM_INSTRUCTION,
            temperature=0.2,
        ),
    )
    
    return response.text
