"""`index/exif.py` — Phase 2 (PLAN.md §6 Phase 2): extract GPS coordinates
from attachment image files and reverse-geocode them offline into a place
name string, for `attachment_place` (the one genuinely non-rebuildable
table in `index.db` -- see PLAN.md §7's backup note -- because it needs
the original attachment files still present on disk).

Invoked *during chunk construction* (from `index/build.py`), not as a
separate pass: for each attachment referenced by messages in the chunk
currently being built, look it up if not already in `attachment_place`,
and feed the result into `render.format_lexical`'s `places_by_attachment`
argument so each geotagged photo's place name appears inline at the
position its message occurs.

**Deviation from PLAN.md's suggested `exifread`/`pillow` pairing**: iOS
photos shared in Messages are overwhelmingly HEIC (confirmed against
real data -- a scan of 500 recent image attachments found essentially
all were `.heic`/`.HEIC`), and neither stock Pillow nor `exifread` can
decode HEIC at all. `pillow-heif` (added as a dependency; not originally
listed in PLAN.md's dependency table) registers a HEIF opener for
Pillow, after which Pillow's own `Image.getexif()` /
`get_ifd(ExifTags.IFD.GPSInfo)` handles HEIC, JPEG, and PNG uniformly
through one code path -- so this module doesn't use `exifread` at all,
despite it remaining an installed dependency (also used by nothing else
in the codebase; harmless to keep, cheap to drop later if it's cleaned
up in a future pass).

**Validated against real data**: of 500 recent real image attachments
checked, ~17% carried GPS EXIF tags (the rest are either not
geotagged, e.g. screenshots/memes, or had Location Services off at
capture time) -- `reverse_geocoder`'s offline GeoNames-derived lookup
correctly resolved real coordinates to real city/region names (e.g. a
[REDACTED_LOCATION] coordinate -> "[REDACTED_LOCATION]").
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pillow_heif
import reverse_geocoder as rg
from PIL import ExifTags, Image, UnidentifiedImageError

pillow_heif.register_heif_opener()

# Small, deliberately incomplete ISO-3166 alpha-2 -> country name map,
# covering the countries most likely to show up in a personal photo
# library. reverse_geocoder only returns the alpha-2 code (`cc`); an
# unmapped code falls back to the raw code itself rather than failing --
# still a usable (if less pretty) lexical signal. Not using a `pycountry`
# dependency for this -- a fixed small map avoids pulling in a full ISO
# country database for a feature this narrow.
_COUNTRY_NAMES: Dict[str, str] = {
    "US": "United States",
    "GB": "United Kingdom",
    "CA": "Canada",
    "PT": "Portugal",
    "ES": "Spain",
    "FR": "France",
    "IT": "Italy",
    "DE": "Germany",
    "MX": "Mexico",
    "JP": "Japan",
    "AU": "Australia",
    "NL": "Netherlands",
    "IE": "Ireland",
    "CH": "Switzerland",
    "BR": "Brazil",
}


@dataclasses.dataclass(frozen=True)
class AttachmentTarget:
    attachment_id: int
    path: Path


def _dms_to_decimal(dms: Tuple[float, float, float], ref: str) -> float:
    """Convert EXIF GPS's (degrees, minutes, seconds) + hemisphere ref
    ('N'/'S'/'E'/'W') into a signed decimal-degree float.
    """
    degrees, minutes, seconds = dms
    value = degrees + minutes / 60.0 + seconds / 3600.0
    if ref in ("S", "W"):
        value = -value
    return value


def extract_gps(path: Path) -> Optional[Tuple[float, float]]:
    """Open an image file (HEIC/JPEG/PNG, via Pillow + pillow-heif) and
    return its (latitude, longitude) if present, else None. Raises
    nothing on a missing/corrupt/unreadable file -- returns None, same
    as "no GPS tag" -- because from the caller's perspective (deciding
    whether to write an `attachment_place` row) those are the same
    outcome. Use `extract_places_for_attachments` if you need to
    distinguish "no GPS" from "couldn't even open the file" for
    `attachment_retry` bookkeeping.
    """
    try:
        image = Image.open(path)
        exif = image.getexif()
        gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
    except (UnidentifiedImageError, OSError, ValueError, KeyError):
        return None
    if not gps_ifd:
        return None

    lat_dms = gps_ifd.get(2)
    lat_ref = gps_ifd.get(1)
    lon_dms = gps_ifd.get(4)
    lon_ref = gps_ifd.get(3)
    if not (lat_dms and lat_ref and lon_dms and lon_ref):
        return None

    try:
        lat = _dms_to_decimal(tuple(float(x) for x in lat_dms), lat_ref)
        lon = _dms_to_decimal(tuple(float(x) for x in lon_dms), lon_ref)
    except (TypeError, ValueError):
        return None
    return lat, lon


def _place_string(geocode_result: dict) -> str:
    country = _COUNTRY_NAMES.get(geocode_result["cc"], geocode_result["cc"])
    parts = [geocode_result.get("name"), geocode_result.get("admin1"), country]
    return " ".join(p for p in parts if p)


def extract_places_for_attachments(
    targets: Sequence[AttachmentTarget],
) -> Tuple[Dict[int, str], List[int]]:
    """For each attachment target, try to resolve a GPS-derived place
    name. Returns `(places, failed_attachment_ids)`:

    - `places`: attachment_id -> "City Admin1 Country" for every
      attachment that had a readable, geotagged image. Attachments that
      opened fine but simply carry no GPS tag are silently absent here
      -- that is the expected, common case (screenshots, memes,
      Location Services off), not a failure.
    - `failed_attachment_ids`: attachment ids whose file could not be
      opened/decoded at all (missing on disk, corrupt, unsupported
      format) -- candidates for `attachment_retry` per PLAN.md §6 Phase 2,
      so a later pass (once the file reappears, e.g. after iCloud
      download finishes) can retry them instead of silently losing the
      signal forever.

    Reverse-geocoding is batched into one `reverse_geocoder.search()`
    call across every resolved coordinate, rather than one call per
    attachment -- `reverse_geocoder` loads a multi-hundred-MB in-memory
    k-d tree of the world's cities on first use; batching amortises
    that (and the per-call lookup itself) across every attachment in the
    current chunk-construction batch instead of paying any part of it
    once per photo.
    """
    coords: List[Tuple[float, float]] = []
    coord_attachment_ids: List[int] = []
    failed: List[int] = []

    for target in targets:
        if not target.path.exists():
            failed.append(target.attachment_id)
            continue
        gps = extract_gps(target.path)
        if gps is None:
            # Distinguish "opened fine, no GPS tag" (common, not a
            # failure) from "couldn't open at all" (retry-worthy) by
            # re-checking whether the file at least opens as an image.
            try:
                Image.open(target.path).getexif()
            except (UnidentifiedImageError, OSError):
                failed.append(target.attachment_id)
            continue
        coords.append(gps)
        coord_attachment_ids.append(target.attachment_id)

    if not coords:
        return {}, failed

    results = rg.search(coords)
    places = {
        attachment_id: _place_string(result)
        for attachment_id, result in zip(coord_attachment_ids, results)
    }
    return places, failed
