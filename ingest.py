import argparse
from pathlib import Path
import sys
from core.database import SessionLocal, init_db
from core.models import ImagingSession
from core.rag import build_document_text, generate_embedding
from ingestion.enrichment import get_sky_and_weather_context
from ingestion.exif import extract_exif, strip_exif_to_bytes
from ingestion.storage import upload_sanitized_image

def parse_args() -> argparse.Namespace:
    parser=argparse.ArgumentParser(
        description="AstroLog CLI: ingest an astrophotography image with automated EXIF stripping, sky ephemeris, weather enrichment, and pgvector indexing."
    )
    parser.add_argument(
        "image_path",
        type=Path,
        help="Path to the local JPEG image file (e.g., photos/DSC_0104.JPG)",
    )
    parser.add_argument(
        "--target",
        required=True,
        type=str,
        help="Celestial target name (e.g., 'Moon', 'Saturn', 'Jupiter')",
    )
    parser.add_argument(
        "--rating",
        required=True,
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Session quality rating from 1 (poor) to 5 (excellent)",
    )
    parser.add_argument(
        "--optics",
        type=str,
        default="Starfield 8-inch Dobsonian + SVBONY 2x Barlow T-Adapter",
        help="Optical train description (defaults to Starfield 8-inch Dobsonian + 2x Barlow)",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Observer notes on seeing conditions, focus, or surface features",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    if not args.image_path.exists():
        print(f"Error: File not found at '{args.image_path}'", file=sys.stderr)
        sys.exit(1)

    print(f"1/5 Extracting EXIF metadata from '{args.image_path.name}'...")
    exif = extract_exif(args.image_path)
    print(
        f"    -> Captured: {exif.captured_at_utc.strftime('%Y-%m-%d %H:%M:%S UTC')} | "
        f"Camera: {exif.camera_model} | ISO: {exif.iso} | Shutter: {exif.shutter_speed}"
    )

    print("2/5 Stripping EXIF tags in memory and uploading sanitized JPEG to Cloudflare R2...")
    clean_bytes = strip_exif_to_bytes(args.image_path)
    image_url, object_key = upload_sanitized_image(
        image_bytes=clean_bytes,
        target_name=args.target,
        captured_at_utc=exif.captured_at_utc,
    )
    print(f"    -> Uploaded to R2: {image_url}")

    print(f"3/5 Computing Astropy ephemeris & Open-Meteo historical weather for '{args.target}'...")
    context = get_sky_and_weather_context(
        target_name=args.target,
        captured_at_utc=exif.captured_at_utc,
    )
    print(
        f"    -> Alt: {context.target_altitude_deg}°, Az: {context.target_azimuth_deg}°, "
        f"Moon Illum: {context.moon_illumination_pct}% | "
        f"Clouds: {context.cloud_cover_pct}%, Wind: {context.wind_speed_kmh} km/h, Temp: {context.temperature_c}°C"
    )

    print("4/5 Synthesizing RAG document and generating Gemini 3072-dimensional vector embedding...")
    doc_text = build_document_text(
        target_name=args.target,
        captured_at_utc=exif.captured_at_utc,
        optics_description=args.optics,
        rating=args.rating,
        camera_model=exif.camera_model,
        iso=exif.iso,
        shutter_speed=exif.shutter_speed,
        target_altitude_deg=context.target_altitude_deg,
        target_azimuth_deg=context.target_azimuth_deg,
        moon_illumination_pct=context.moon_illumination_pct,
        cloud_cover_pct=context.cloud_cover_pct,
        wind_speed_kmh=context.wind_speed_kmh,
        temperature_c=context.temperature_c,
        user_notes=args.notes,
    )
    embedding = generate_embedding(doc_text)

    print("5/5 Saving session record and vector embedding to Neon PostgreSQL...")
    init_db()
    with SessionLocal() as db:
        session_record = ImagingSession(
            target_name=args.target.strip(),
            optics_description=args.optics.strip(),
            rating=args.rating,
            user_notes=args.notes.strip() or None,
            image_url=image_url,
            object_key=object_key,
            captured_at=exif.captured_at_utc,
            camera_model=exif.camera_model,
            iso=exif.iso,
            shutter_speed=exif.shutter_speed,
            target_altitude_deg=context.target_altitude_deg,
            target_azimuth_deg=context.target_azimuth_deg,
            moon_illumination_pct=context.moon_illumination_pct,
            cloud_cover_pct=context.cloud_cover_pct,
            wind_speed_kmh=context.wind_speed_kmh,
            temperature_c=context.temperature_c,
            document_text=doc_text,
            embedding=embedding,
        )
        db.add(session_record)
        db.commit()
        db.refresh(session_record)
        print(f"\nSuccess! Logged Session #{session_record.id} ({session_record.target_name}) to AstroLog.")

if __name__ == "__main__":
    main()
