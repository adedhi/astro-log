from datetime import datetime, timezone
from core.rag import build_document_text
from ingestion.enrichment import compute_sky_context

def test_compute_sky_context_returns_valid_ranges() -> None:
    captured = datetime(2026, 10, 15, 2, 0, tzinfo=timezone.utc)
    alt_deg, az_deg, moon_illum_pct = compute_sky_context("Saturn", captured)

    assert -90.0 <= alt_deg <= 90.0
    assert 0.0 <= az_deg <= 360.0
    assert 0.0 <= moon_illum_pct <= 100.0

def test_build_document_text_includes_all_telemetry() -> None:
    captured = datetime(2026, 10, 15, 2, 0, tzinfo=timezone.utc)
    doc = build_document_text(
        target_name="Moon",
        captured_at_utc=captured,
        optics_description="Starfield 8-inch Dobsonian + 2x Barlow",
        rating=4,
        camera_model="NIKON D5200",
        iso=200,
        shutter_speed="1/250s",
        target_altitude_deg=45.2,
        target_azimuth_deg=180.5,
        moon_illumination_pct=68.4,
        cloud_cover_pct=10,
        wind_speed_kmh=12.0,
        temperature_c=14.5,
        user_notes="Crisp craters along the terminator.",
    )

    assert "Target: Moon" in doc
    assert "NIKON D5200 (ISO 200, Shutter 1/250s)" in doc
    assert "Target Altitude: 45.2°" in doc
    assert "Crisp craters along the terminator." in doc
