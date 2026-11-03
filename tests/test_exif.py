import io
from pathlib import Path
from PIL import ExifTags, Image
from ingestion.exif import _format_shutter_speed, extract_exif, strip_exif_to_bytes

def _create_jpeg_with_exif(tmp_path: Path) -> Path:
    """Creates a temporary JPEG containing Nikon D5200 EXIF & GPS tags."""
    img = Image.new("RGB", (64, 64), color=(20, 40, 80))
    exif = Image.Exif()
    exif[ExifTags.Base.Model] = "NIKON D5200"
    exif[ExifTags.Base.DateTime] = "2026:10:15 22:30:00"

    # Populated Exif Sub-IFD (Image File Directory)
    exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
    exif_ifd[ExifTags.Base.DateTimeOriginal] = "2026:10:15 22:30:00"
    exif_ifd[ExifTags.Base.ISOSpeedRatings] = 400
    exif_ifd[ExifTags.Base.ExposureTime] = 0.004  # 1/250s

    file_path = tmp_path / "test_moon.jpg"
    img.save(file_path, format="JPEG", exif=exif.tobytes())
    return file_path

def test_format_shutter_speed() -> None:
    assert _format_shutter_speed(0.004) == "1/250s"
    assert _format_shutter_speed(0.008) == "1/125s"
    assert _format_shutter_speed(1.5) == "1.5s"
    assert _format_shutter_speed(None) is None

def test_extract_exif_parses_fields_and_converts_to_utc(tmp_path: Path) -> None:
    jpeg_path = _create_jpeg_with_exif(tmp_path)
    metadata = extract_exif(jpeg_path)

    assert metadata.camera_model == "NIKON D5200"
    assert metadata.iso == 400
    assert metadata.shutter_speed == "1/250s"
    assert metadata.captured_at_utc.tzinfo is not None
    assert metadata.captured_at_utc.utcoffset().total_seconds() == 0

def test_strip_exif_removes_all_metadata(tmp_path: Path) -> None:
    jpeg_path = _create_jpeg_with_exif(tmp_path)
    sanitized_bytes = strip_exif_to_bytes(jpeg_path)

    with Image.open(io.BytesIO(sanitized_bytes)) as sanitized_img:
        sanitized_exif = sanitized_img.getexif()
        assert len(sanitized_exif) == 0
