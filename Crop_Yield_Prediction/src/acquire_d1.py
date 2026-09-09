"""Acquires D1 Government Crop Statistics via official data.gov.in API.

Features:
- Reads GOV_API_KEY from environment.
- Bounded retries with exponential backoff.
- Pagination using offset and limit.
- Cycle/infinite loop detection.
- Detailed acquisition provenance saved without exposing secrets.
"""

import argparse
import datetime
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import requests

try:
    from src.config import (
        API_MAX_RETRIES,
        API_PAGE_LIMIT,
        API_RETRY_BACKOFF_FACTOR,
        API_TIMEOUT_SECONDS,
        D1_RAW_FILE,
        DEFAULT_API_URL,
        GOV_API_URL,
        PROVENANCE_DIR,
    )
    from src.utils import compute_sha256, mask_secret, save_json, setup_logger
except ImportError:
    from config import (
        API_MAX_RETRIES,
        API_PAGE_LIMIT,
        API_RETRY_BACKOFF_FACTOR,
        API_TIMEOUT_SECONDS,
        D1_RAW_FILE,
        DEFAULT_API_URL,
        GOV_API_URL,
        PROVENANCE_DIR,
    )
    from utils import compute_sha256, mask_secret, save_json, setup_logger

logger = setup_logger("acquire_d1")


def get_api_key() -> str:
    """Retrieves API key from GOV_API_KEY environment variables.

    Raises:
        ValueError: If neither environment variable is found.
    """
    key = os.getenv("GOV_API_KEY")
    if not key or not key.strip():
        raise ValueError(
            "API key not found! Please ensure GOV_API_KEY is configured in your .env file or environment."
        )
    return key.strip()


def fetch_page_with_retry(
    url: str,
    params: Dict[str, Any],
    max_retries: int = API_MAX_RETRIES,
    timeout: int = API_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Fetches a single page with bounded retries and exponential backoff."""
    headers = {"User-Agent": "Agricultural-Predictive-Analytics/1.0 (Research Pipeline)"}
    delay = 1.0

    for attempt in range(1, max_retries + 1):
        try:
            logger.debug("Requesting offset %s (Attempt %d/%d)", params.get("offset"), attempt, max_retries)
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            if response.status_code == 200:
                return response.json()
            elif response.status_code in [429, 500, 502, 503, 504]:
                logger.warning(
                    "HTTP %d encountered on attempt %d/%d. Backing off for %.1fs...",
                    response.status_code,
                    attempt,
                    max_retries,
                    delay,
                )
                time.sleep(delay)
                delay *= API_RETRY_BACKOFF_FACTOR
            else:
                response.raise_for_status()
        except (requests.RequestException, requests.Timeout) as exc:
            logger.warning(
                "Network error on attempt %d/%d: %s. Backing off for %.1fs...",
                attempt,
                max_retries,
                str(exc),
                delay,
            )
            if attempt == max_retries:
                raise RuntimeError(
                    f"Failed to fetch data after {max_retries} attempts: {exc}"
                ) from exc
            time.sleep(delay)
            delay *= API_RETRY_BACKOFF_FACTOR

    raise RuntimeError(f"Failed to fetch from {url} after {max_retries} retries.")


def acquire_d1(
    output_file: Path = D1_RAW_FILE,
    crop_filter: Optional[str] = None,
    limit: int = API_PAGE_LIMIT,
    force: bool = False,
) -> Path:
    """Acquires D1 dataset via official API and saves raw CSV and provenance metadata.

    Args:
        output_file: Destination file path for raw CSV.
        crop_filter: Optional filter for specific crop (e.g. 'Rice').
        limit: Records per request page.
        force: Overwrite existing raw file if True.

    Returns:
        Path to the saved raw CSV.
    """
    if output_file.exists() and not force:
        logger.info("Raw D1 file already exists at %s. Use --force to re-acquire.", output_file)
        return output_file

    api_key = get_api_key()
    logger.info("Validated API key present: %s", mask_secret(api_key))

    url = GOV_API_URL or DEFAULT_API_URL
    logger.info("Target API endpoint: %s", url)

    all_records: List[Dict[str, Any]] = []
    offset = 0
    page_count = 0
    total_expected: Optional[int] = None
    seen_page_first_records = set()

    base_params: Dict[str, Any] = {
        "api-key": api_key,
        "format": "json",
        "limit": limit,
    }
    if crop_filter:
        base_params["filters[crop]"] = crop_filter

    t_start = time.time()
    logger.info("Beginning data acquisition with page limit=%d...", limit)

    while True:
        params = dict(base_params)
        params["offset"] = offset

        payload = fetch_page_with_retry(url, params)
        records = payload.get("records", [])
        page_count += 1

        if total_expected is None and "total" in payload:
            total_expected = int(payload["total"])
            logger.info("Total available records reported by API: %d", total_expected)

        if not records:
            logger.info("Page %d returned 0 records. End of pagination reached.", page_count)
            break

        # Infinite loop safeguard: verify page fingerprint
        first_record_repr = repr(records[0])
        if first_record_repr in seen_page_first_records:
            logger.warning("Repeated page detected at offset %d. Terminating loop.", offset)
            break
        seen_page_first_records.add(first_record_repr)

        all_records.extend(records)
        logger.info(
            "Page %d fetched (%d records). Cumulative total: %d / %s",
            page_count,
            len(records),
            len(all_records),
            total_expected or "unknown",
        )

        if total_expected and len(all_records) >= total_expected:
            logger.info("All %d expected records acquired.", total_expected)
            break

        offset += len(records)
        # Small courteous delay between pagination requests
        time.sleep(0.1)

    duration = time.time() - t_start
    logger.info(
        "Acquisition complete. Acquired %d records across %d pages in %.2f seconds.",
        len(all_records),
        page_count,
        duration,
    )

    if not all_records:
        raise RuntimeError("No records were fetched from API. Verify network and query parameters.")

    # Convert to DataFrame and save raw CSV
    df = pd.DataFrame(all_records)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, encoding="utf-8")
    logger.info("Raw data saved to %s (%d rows, %d columns)", output_file, len(df), len(df.columns))

    # Calculate SHA-256
    file_sha256 = compute_sha256(output_file)

    # Compile provenance metadata (NEVER log the API key)
    safe_params = {k: v for k, v in base_params.items() if k != "api-key"}
    safe_params["final_offset"] = offset

    provenance = {
        "dataset_id": "D1_government_crop_production_statistics",
        "title": "District-wise, season-wise crop production statistics from 1997",
        "source_url": url,
        "endpoint": url,
        "request_parameters": safe_params,
        "download_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "number_of_records": len(df),
        "number_of_pages": page_count,
        "raw_file": str(output_file.name),
        "raw_sha256": file_sha256,
        "columns_received": list(df.columns),
        "http_status": "200 OK",
        "source_publisher": "Ministry of Agriculture and Farmers Welfare, Government of India",
        "portal": "Open Government Data (OGD) Platform India (data.gov.in)",
        "version_or_release": "1997-2015 published historical series",
        "licence_audit": {
            "licence_status": "unresolved",
            "note": "Open Government Data (OGD) Platform India - National Data Sharing and Accessibility Policy (NDSAP). Explicit redistribution terms require verification.",
        },
        "geographic_coverage": "All participating States and Union Territories of India",
        "temporal_coverage": "1997-2015",
        "target_unit": "t/ha (derived: production Tonnes / area Hectares)",
        "responsible_reviewer": "MDI3003 Research Pipeline Auditor",
    }

    provenance_file = PROVENANCE_DIR / "d1_acquisition.json"
    save_json(provenance, provenance_file)
    logger.info("Acquisition provenance saved to %s", provenance_file)

    return output_file


def main():
    parser = argparse.ArgumentParser(description="Acquire D1 crop statistics from data.gov.in")
    parser.add_argument(
        "--crop",
        type=str,
        default=None,
        help="Optional API-level crop filter (e.g., 'Rice'). Default acquires complete dataset.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=API_PAGE_LIMIT,
        help=f"Records per page (default: {API_PAGE_LIMIT})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-acquisition even if raw file already exists.",
    )
    args = parser.parse_args()

    try:
        acquire_d1(crop_filter=args.crop, limit=args.limit, force=args.force)
    except Exception as e:
        logger.error("Acquisition failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
