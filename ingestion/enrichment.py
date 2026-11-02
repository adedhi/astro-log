from dataclasses import dataclass
from datetime import datetime, timezone
import astropy.units as u
import httpx
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, get_body
from astropy.time import Time
from core.config import settings

@dataclass
class SkyWeatherContext:
    target_altitude_deg: float
    target_azimuth_deg: float
    moon_illumination_pct: float
    cloud_cover_pct: int | None
    wind_speed_kmh: float | None
    temperature_c: float | None

def compute_sky_context(target_name: str, captured_at_utc: datetime) -> tuple[float, float, float]:
    """
    Computes target altitude (deg), azimuth (deg), and lunar illumination (%)
    at the exact UTC capture timestamp from the observer's coordinates.
    """
    obs_time = Time(captured_at_utc)
    location = EarthLocation(
        lat=settings.OBSERVER_LATITUDE * u.deg,
        lon=settings.OBSERVER_LONGITUDE * u.deg,
        height=settings.OBSERVER_ELEVATION_M * u.m,
    )
    altaz_frame = AltAz(obstime=obs_time, location=location)

    # Calculate target Altitude and Azimuth (supports 'moon', 'saturn', 'jupiter', 'mars', 'venus', etc.)
    body_key = target_name.strip().lower()
    try:
        target_coord = get_body(body_key, obs_time, location)
    except Exception as exc:
        raise ValueError(
            f"Astropy could not resolve solar system body '{target_name}'. "
            f"Use a standard body name like 'Moon', 'Saturn', 'Jupiter', 'Mars', or 'Venus'."
        ) from exc

    target_altaz = target_coord.transform_to(altaz_frame)
    alt_deg = round(float(target_altaz.alt.deg), 2)
    az_deg = round(float(target_altaz.az.deg), 2)

    # Calculate Moon Illumination percentage via Sun-Moon elongation angle
    sun_coord = get_body("sun", obs_time, location)
    moon_coord = get_body("moon", obs_time, location)
    elongation = sun_coord.separation(moon_coord)
    moon_illum_pct = round(float((1.0 - np.cos(elongation.rad)) / 2.0 * 100.0), 1)

    return alt_deg, az_deg, moon_illum_pct

def fetch_historical_weather(captured_at_utc: datetime) -> tuple[int | None, float | None, float | None]:
    """
    Fetches hourly cloud cover (%), wind speed (km/h), and temperature (°C)
    from Open-Meteo for the UTC hour closest to captured_at_utc.
    """
    now_utc = datetime.now(timezone.utc)
    days_ago = (now_utc - captured_at_utc).total_seconds() / 86400.0
    date_str = captured_at_utc.strftime("%Y-%m-%d")

    # Open-Meteo's archive endpoint has a ~2-5 day lag; use forecast endpoint for recent session dates
    if days_ago <= 5.0:
        url = "https://api.open-meteo.com/v1/forecast"
    else:
        url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": settings.OBSERVER_LATITUDE,
        "longitude": settings.OBSERVER_LONGITUDE,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": "temperature_2m,cloud_cover,wind_speed_10m",
        "timezone": "UTC",
    }

    with httpx.Client(timeout=10.0) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        data = response.json()

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    target_hour_prefix = captured_at_utc.strftime("%Y-%m-%dT%H:00")

    if target_hour_prefix not in times:
        return None, None, None

    index = times.index(target_hour_prefix)
    cloud_cover = hourly.get("cloud_cover", [None])[index]
    wind_speed = hourly.get("wind_speed_10m", [None])[index]
    temperature = hourly.get("temperature_2m", [None])[index]

    return (
        int(round(cloud_cover)) if cloud_cover is not None else None,
        round(float(wind_speed), 1) if wind_speed is not None else None,
        round(float(temperature), 1) if temperature is not None else None,
    )

def get_sky_and_weather_context(target_name: str, captured_at_utc: datetime) -> SkyWeatherContext:
    """Combines Astropy sky ephemeris and Open-Meteo weather into a single SkyWeatherContext object."""
    alt_deg, az_deg, moon_illum_pct = compute_sky_context(target_name, captured_at_utc)
    cloud_cover, wind_speed, temperature = fetch_historical_weather(captured_at_utc)

    return SkyWeatherContext(
        target_altitude_deg=alt_deg,
        target_azimuth_deg=az_deg,
        moon_illumination_pct=moon_illum_pct,
        cloud_cover_pct=cloud_cover,
        wind_speed_kmh=wind_speed,
        temperature_c=temperature,
    )
