"""Unit tests for seaglass.index.exif -- Phase 2 GPS extraction +
offline reverse-geocoding for attachment_place. Synthetic JPEGs with
hand-written GPS EXIF tags (no network dependency; reverse_geocoder's
own gazetteer is bundled/offline per its package data, no HTTP calls).
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from PIL.ExifTags import IFD

from seaglass.index.exif import (
    AttachmentTarget,
    _dms_to_decimal,
    extract_gps,
    extract_places_for_attachments,
)


def _write_geotagged_jpeg(path: Path, lat: float, lon: float) -> None:
    """Write a tiny synthetic JPEG with GPS EXIF tags for (lat, lon),
    using the same DMS + hemisphere-ref encoding real cameras/phones use.
    """
    lat_ref = "N" if lat >= 0 else "S"
    lon_ref = "E" if lon >= 0 else "W"
    lat, lon = abs(lat), abs(lon)

    def to_dms(value: float):
        degrees = int(value)
        minutes_float = (value - degrees) * 60
        minutes = int(minutes_float)
        seconds = (minutes_float - minutes) * 60
        return (float(degrees), float(minutes), float(seconds))

    img = Image.new("RGB", (4, 4), color="blue")
    exif = img.getexif()
    gps_ifd = exif.get_ifd(IFD.GPSInfo)
    gps_ifd[1] = lat_ref
    gps_ifd[2] = to_dms(lat)
    gps_ifd[3] = lon_ref
    gps_ifd[4] = to_dms(lon)
    exif[IFD.GPSInfo] = gps_ifd
    img.save(path, exif=exif)


def _write_plain_jpeg(path: Path) -> None:
    Image.new("RGB", (4, 4), color="green").save(path)


def test_dms_to_decimal_handles_all_hemispheres():
    assert abs(_dms_to_decimal((37.0, 46.0, 26.4), "N") - 37.774) < 1e-3
    assert _dms_to_decimal((37.0, 46.0, 26.4), "S") < 0
    assert _dms_to_decimal((122.0, 25.0, 9.6), "E") > 0
    assert _dms_to_decimal((122.0, 25.0, 9.6), "W") < 0


def test_extract_gps_round_trips_synthetic_coordinates(tmp_path):
    path = tmp_path / "geotagged.jpg"
    _write_geotagged_jpeg(path, 37.7749, -122.4194)  # San Francisco

    result = extract_gps(path)

    assert result is not None
    lat, lon = result
    assert abs(lat - 37.7749) < 1e-3
    assert abs(lon - (-122.4194)) < 1e-3


def test_extract_gps_returns_none_for_untagged_image(tmp_path):
    path = tmp_path / "plain.jpg"
    _write_plain_jpeg(path)

    assert extract_gps(path) is None


def test_extract_gps_returns_none_for_missing_file(tmp_path):
    assert extract_gps(tmp_path / "does_not_exist.jpg") is None


def test_extract_gps_returns_none_for_corrupt_file(tmp_path):
    path = tmp_path / "corrupt.jpg"
    path.write_bytes(b"not actually a jpeg")

    assert extract_gps(path) is None


def test_extract_places_for_attachments_resolves_geotagged_images(tmp_path):
    geotagged = tmp_path / "geotagged.jpg"
    _write_geotagged_jpeg(geotagged, 37.7749, -122.4194)  # San Francisco
    plain = tmp_path / "plain.jpg"
    _write_plain_jpeg(plain)

    targets = [
        AttachmentTarget(attachment_id=1, path=geotagged),
        AttachmentTarget(attachment_id=2, path=plain),
    ]
    places, failed = extract_places_for_attachments(targets)

    assert 1 in places
    assert "United States" in places[1] or "US" in places[1]
    assert 2 not in places  # untagged, not a failure
    assert failed == []


def test_extract_places_for_attachments_flags_missing_file_as_failed(tmp_path):
    targets = [AttachmentTarget(attachment_id=99, path=tmp_path / "gone.jpg")]

    places, failed = extract_places_for_attachments(targets)

    assert places == {}
    assert failed == [99]


def test_extract_places_for_attachments_flags_corrupt_file_as_failed(tmp_path):
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"not actually a jpeg")
    targets = [AttachmentTarget(attachment_id=7, path=corrupt)]

    places, failed = extract_places_for_attachments(targets)

    assert places == {}
    assert failed == [7]


def test_extract_places_for_attachments_empty_input():
    places, failed = extract_places_for_attachments([])
    assert places == {}
    assert failed == []
