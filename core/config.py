from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Required by both ingest.py and app.py
    DATABASE_URL: str
    GEMINI_API_KEY: str

    # Required only by ingest.py (Cloudflare R2 upload)
    R2_ENDPOINT_URL: str = ""
    R2_ACCESS_KEY_ID: str = ""
    R2_SECRET_ACCESS_KEY: str = ""
    R2_BUCKET_NAME: str = ""
    R2_PUBLIC_URL: str = ""

    # Required only by ingest.py (local observer coordinates)
    OBSERVER_LATITUDE: float = 0.0
    OBSERVER_LONGITUDE: float = 0.0
    OBSERVER_ELEVATION_M: float = 0.0
    OBSERVER_TIMEZONE: str = "America/Toronto"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()
