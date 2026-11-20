#!/usr/bin/env python3
"""Upload or sync presentation slides directly to Google Slides using the Google Slides API.

By replacing slides with rendered PowerPoint pages, this workflow:
- Avoids the Google Drive Service Account "storage quota exceeded" error.
- Updates the exact existing Google Slides presentation in-place (URL never changes).
- Keeps the presentation URL but removes its old slides and their contents.

The resulting Google Slides pages are images. Text, shapes, animations and
speaker notes from the PowerPoint are not individually editable there.
"""

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/presentations",
]


def replacement_requests(old_slides, image_urls, run_id=None):
    """Build one atomic replacement, creating pages before deleting old ones."""
    if not image_urls:
        raise ValueError("Refusing to replace a presentation with zero slides")
    run_id = run_id or uuid.uuid4().hex
    requests = []
    for i, url in enumerate(image_urls):
        slide_id = f"sync_{run_id}_{i:03d}"
        requests.extend([
            {"createSlide": {"objectId": slide_id,
                             "slideLayoutReference": {"predefinedLayout": "BLANK"}}},
            {"updatePageProperties": {
                "objectId": slide_id,
                "pageProperties": {"pageBackgroundFill": {
                    "stretchedPictureFill": {"contentUrl": url}}},
                "fields": "pageBackgroundFill",
            }},
        ])
    requests.extend({"deleteObject": {"objectId": slide["objectId"]}}
                    for slide in old_slides)
    return requests


def get_credentials(credentials_path: str | None = None):
    """Authenticate with Google Cloud using a Service Account."""
    from google.oauth2 import service_account

    if credentials_path and os.path.isfile(credentials_path):
        return service_account.Credentials.from_service_account_file(
            credentials_path, scopes=SCOPES
        )

    env_key = os.environ.get("GCP_SA_KEY")
    if env_key:
        env_key = env_key.strip()
        if os.path.isfile(env_key):
            return service_account.Credentials.from_service_account_file(
                env_key, scopes=SCOPES
            )
        try:
            key_info = json.loads(env_key)
            return service_account.Credentials.from_service_account_info(
                key_info, scopes=SCOPES
            )
        except json.JSONDecodeError as e:
            raise ValueError(
                "GCP_SA_KEY is set but does not appear to be valid JSON or a file path."
            ) from e

    google_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if google_creds and os.path.isfile(google_creds):
        return service_account.Credentials.from_service_account_file(
            google_creds, scopes=SCOPES
        )

    raise RuntimeError(
        "No Google Cloud Service Account credentials found. "
        "Set the GCP_SA_KEY secret/env var or provide --credentials."
    )


def sync_to_google_slides(
    creds,
    presentation_id: str,
    image_urls: list[str],
):
    """Sync slide images into an existing Google Slides deck via the Slides API."""
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    try:
        slides_service = build("slides", "v1", credentials=creds)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Google Slides service: {e}") from e

    print(f"Connecting to Google Slides presentation ID: {presentation_id}...")
    try:
        pres = (
            slides_service.presentations()
            .get(presentationId=presentation_id)
            .execute()
        )
    except HttpError as e:
        err_msg = str(e)
        if "SERVICE_DISABLED" in err_msg or "slides.googleapis.com" in err_msg:
            print(
                "\n" + "=" * 70 + "\n"
                "ACTION REQUIRED: Google Slides API is not enabled in your Google Cloud Project!\n\n"
                "Please open this link in your browser and click 'ENABLE':\n"
                "https://console.cloud.google.com/apis/library/slides.googleapis.com\n"
                + "=" * 70 + "\n",
                file=sys.stderr,
            )
            sys.exit(1)
        elif e.resp.status == 404:
            print(
                f"\nERROR: Presentation '{presentation_id}' not found.\n"
                "Ensure the presentation ID is correct and shared with the Service Account email.\n",
                file=sys.stderr,
            )
            sys.exit(1)
        raise

    title = pres.get("title", "Untitled Presentation")
    old_slides = pres.get("slides", [])
    print(f"Connected to '{title}'. Current deck has {len(old_slides)} slide(s).")
    print(f"Preparing to sync {len(image_urls)} new slide(s)...")

    requests = replacement_requests(old_slides, image_urls)
    body = {"requests": requests}
    if pres.get("revisionId"):
        body["writeControl"] = {"requiredRevisionId": pres["revisionId"]}
    print("Replacing all old slides in one batchUpdate...")
    slides_service.presentations().batchUpdate(
        presentationId=presentation_id, body=body
    ).execute()

    view_link = f"https://docs.google.com/presentation/d/{presentation_id}/edit"
    print(f"\nSUCCESS! Synced {len(image_urls)} slide(s) to Google Slides:")
    print(f"-> {view_link}\n")
    return view_link


def main():
    parser = argparse.ArgumentParser(
        description="Sync presentation slides directly to Google Slides deck"
    )
    parser.add_argument(
        "--presentation-id",
        type=str,
        default=os.environ.get("GDRIVE_PRESENTATION_ID")
        or os.environ.get("GDRIVE_FILE_ID"),
        help="Google Slides Presentation ID (from the presentation URL)",
    )
    parser.add_argument(
        "--slides-dir",
        type=Path,
        default=Path("presentation/slides"),
        help="Directory containing rendered slide-*.png images",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("SLIDES_BASE_URL"),
        help="Base public URL where slide-*.png images are hosted (e.g. raw.githubusercontent.com)",
    )
    parser.add_argument(
        "--version-tag",
        type=str,
        default=os.environ.get("GITHUB_SHA", "v1"),
        help="Version tag or commit SHA for cache busting",
    )
    parser.add_argument(
        "--credentials",
        type=str,
        default=None,
        help="Path to GCP Service Account JSON key file",
    )

    args = parser.parse_args()

    if not args.presentation_id:
        print(
            "Error: Presentation ID is required. Pass --presentation-id or set GDRIVE_PRESENTATION_ID.",
            file=sys.stderr,
        )
        sys.exit(1)

    slides_dir = args.slides_dir.resolve()
    images = sorted(slides_dir.glob("slide-*.png"))
    if not images:
        images = sorted(slides_dir.glob("page-*.png"))

    if not images:
        print(f"Error: No slide PNG images found in '{slides_dir}'.", file=sys.stderr)
        sys.exit(1)

    if not args.base_url:
        print(
            "Error: --base-url is required to construct public image URLs for Google Slides API.",
            file=sys.stderr,
        )
        sys.exit(1)

    base = args.base_url.rstrip("/")
    image_urls = [f"{base}/{img.name}?v={args.version_tag}" for img in images]

    creds = get_credentials(args.credentials)
    sync_to_google_slides(
        creds=creds,
        presentation_id=args.presentation_id,
        image_urls=image_urls,
    )


if __name__ == "__main__":
    main()
