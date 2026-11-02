from datetime import datetime
from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, DateTime, Float, func, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
    pass

class ImagingSession(Base):
    __tablename__ = "imaging_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # User-provided CLI inputs
    target_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    optics_description: Mapped[str] = mapped_column(Text, nullable=False)
    rating: Mapped[int] = mapped_column(
        Integer, CheckConstraint("rating BETWEEN 1 AND 5"), nullable=False
    )
    user_notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Cloudflare R2 Object Storage (EXIF-stripped image)
    image_url: Mapped[str] = mapped_column(Text, nullable=False)
    object_key: Mapped[str] = mapped_column(Text, nullable=False)

    # Extracted EXIF Metadata
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    camera_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    iso: Mapped[int | None] = mapped_column(Integer, nullable=True)
    shutter_speed: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Automated Sky & Weather Context (Observer Lat/Lon is not stored in DB)
    target_altitude_deg: Mapped[float | None] = mapped_column(Float, nullable=True)
    target_azimuth_deg: Mapped[float | None] = mapped_column(Float, nullable=True)
    moon_illumination_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    cloud_cover_pct: Mapped[int | None] = mapped_column(Integer, nullable=True)
    wind_speed_kmh: Mapped[float | None] = mapped_column(Float, nullable=True)
    temperature_c: Mapped[float | None] = mapped_column(Float, nullable=True)

    # RAG / Vector Search (768 dimensions for Gemini text-embedding-004)
    document_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(768), nullable=False)
