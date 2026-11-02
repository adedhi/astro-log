import boto3
import uuid
from datetime import datetime
from core.config import settings

def upload_sanitized_image(image_bytes: bytes, target_name: str, captured_at_utc: datetime) -> tuple[str, str]:
    """Uploads EXIF-stripped JPEG bytes to Cloudflare R2 and returns (public_url, object_key)."""
    if not all (
        [
            settings.R2_ENDPOINT_URL,
            settings.R2_ACCESS_KEY_ID,
            settings.R2_SECRET_ACCESS_KEY,
            settings.R2_BUCKET_NAME,
            settings.R2_PUBLIC_URL,
        ]
    ):
        raise RuntimeError(
            "Missing Cloudflare R2 credentials in .env. Ensure all R2_* variables are set."
        )


    # Cloudflare R2 is S3-compatible
    s3_client = boto3.client(
        "s3",
        endpoint_url=settings.R2_ENDPOINT_URL,
        aws_access_key_id=settings.R2_ACCESS_KEY_ID,
        aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
        region_name="auto",
    )

    safe_target = "".join(c if c.isalnum() else "_" for c in target_name.lower()).strip("_")
    date_prefix = captured_at_utc.strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:8]
    object_key = f"sessions/{date_prefix}_{safe_target}_{short_id}.jpg"

    s3_client.put_object(
        Bucket=settings.R2_BUCKET_NAME,
        Key=object_key,
        Body=image_bytes,
        ContentType="image/jpeg",
    )

    base_url = settings.R2_PUBLIC_URL.rstrip("/")
    public_url = f"{base_url}/{object_key}"

    return public_url, object_key
