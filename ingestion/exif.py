import io
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from PIL import ExifTags, Image, ImageOps
from core.config import settings

@dataclass
class ExifMetadata:
    captured_at_utc: datetime
    camera_model: str | None
    iso: int | None
    shutter_speed: str | None

def _format_shutter_speed(exposure_time: float | None) -> str | None:
    """Formats a float exposure time (e.g., 0.004) into '1/250s' or '2.5s'."""
    if exposure_time is None or exposure_time <= 0:
        return None
    if exposure_time >= 1.0:
        return f"{round(float(exposure_time), 2):g}s"
    return f"1/{round(1.0 / float(exposure_time))}s"

def extract_exif(image_path: str | Path) -> ExifMetadata:
    """Extracts capture timestamp (converted to UTC), camera model, ISO, and shutter speed."""
    with Image.open(image_path) as img:
        exif_data = img.getexif()
        if not exif_data:
            raise ValueError(f"No EXIF metadata found in {image_path}")

        # Base Image File Directory (IFD) tags (Model = 272, DateTime = 306)
        model = exif_data.get(ExifTags.Base.Model)
        dt_fallback = exif_data.get(ExifTags.Base.DateTime)

        # Exif Sub_IFD tags (DateTimeOriginal = 36867, ExposureTime = 33434, ISO = 34855)
        exif_ifd = exif_data.get_ifd(ExifTags.IFD.Exif)
        dt_str = exif_ifd.get(ExifTags.Base.DateTimeOriginal) or dt_fallback
        if not dt_str:
            raise ValueError(f"No capture timestamp (DateTimeOriginal) found in {image_path}")

        # Parse naive DSLR local time ("YYYY:MM:DD HH:MM:SS") and convert to UTC
        naive_dt = datetime.strptime(str(dt_str).strip(), "%Y:%m:%d %H:%M:%S")
        local_tz = ZoneInfo(settings.OBSERVER_TIMEZONE)
        captured_at_utc = naive_dt.replace(tzinfo=local_tz).astimezone(timezone.utc)

        # Parse DSLR ISO settings
        iso_raw = exif_ifd.get(ExifTags.Base.ISOSpeedRatings)
        iso = int(iso_raw) if iso_raw is not None else None

        # Parse DSLR exposure and format shutter speed setting
        exposure_raw = exif_ifd.get(ExifTags.Base.ExposureTime)
        shutter_speed = (
            _format_shutter_speed(float(exposure_raw)) if exposure_raw is not None else None
        )

        camera_model = str(model).strip() if model else None

        return ExifMetadata(
            captured_at_utc = captured_at_utc,
            camera_model = camera_model,
            iso = iso,
            shutter_speed= shutter_speed,
        )

def strip_exif_to_bytes(image_path: str | Path, quality: int = 92) -> bytes:
    """
    Applies orientation tag to pixel grid, then strips 100% of EXIF/MakerNote/GPS
    metadata (for privacy) by saving raw pixel data into an in-memory JPEG byte buffer.
    """
    with Image.open(image_path) as img:
        # Transform orientation into pixels before stripping tags so portrait shots don't rotate
        oriented_img = ImageOps.exif_transpose(img)
        if oriented_img.mode in ("RGBA", "P"):
            oriented_img = oriented_img.convert("RGB")

        buffer = io.BytesIO()
        oriented_img.save(buffer, format="JPEG", quality=quality)
        return buffer.getvalue()
