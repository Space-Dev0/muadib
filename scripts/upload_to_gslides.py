#!/usr/bin/env python3
"""Upload or sync a PPTX presentation to Google Drive / Google Slides.

Supports:
- Direct update of existing Google Drive presentations.
- Auto-conversion of PPTX to native Google Slides (`application/vnd.google-apps.presentation`).
- Folder-based sync (updating/replacing named presentations in a shared folder).
"""

import argparse
import json
import os
import sys
from pathlib import Path

PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
GSLIDES_MIME = "application/vnd.google-apps.presentation"
SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/presentations",
]


def get_credentials(credentials_path: str | None = None):
    """Authenticate with Google Cloud using a Service Account."""
    from google.oauth2 import service_account

    # 1. Direct path argument
    if credentials_path and os.path.isfile(credentials_path):
        return service_account.Credentials.from_service_account_file(
            credentials_path, scopes=SCOPES
        )

    # 2. GCP_SA_KEY environment variable (raw JSON string or file path)
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
                "GCP_SA_KEY is set but does not appear to be valid JSON or a valid file path."
            ) from e

    # 3. GOOGLE_APPLICATION_CREDENTIALS standard env var
    google_creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if google_creds_path and os.path.isfile(google_creds_path):
        return service_account.Credentials.from_service_account_file(
            google_creds_path, scopes=SCOPES
        )

    raise RuntimeError(
        "No Google Cloud Service Account credentials found. "
        "Set the GCP_SA_KEY secret/env var or provide --credentials."
    )


def build_drive_service(creds):
    """Build and return the Google Drive API service client."""
    from googleapiclient.discovery import build

    return build("drive", "v3", credentials=creds)


def upload_new_presentation(
    drive_service,
    pptx_path: Path,
    title: str,
    folder_id: str | None = None,
    as_gslides: bool = True,
) -> dict:
    """Upload a new presentation, optionally converting to Google Slides."""
    from googleapiclient.http import MediaFileUpload

    file_metadata = {
        "name": title,
    }
    if as_gslides:
        file_metadata["mimeType"] = GSLIDES_MIME

    if folder_id:
        file_metadata["parents"] = [folder_id]

    media = MediaFileUpload(str(pptx_path), mimetype=PPTX_MIME, resumable=True)

    print(f"Uploading '{pptx_path.name}' as '{title}'...")
    uploaded = (
        drive_service.files()
        .create(
            body=file_metadata,
            media_body=media,
            fields="id, name, mimeType, webViewLink",
        )
        .execute()
    )
    return uploaded


def update_pptx_file(drive_service, file_id: str, pptx_path: Path) -> dict:
    """Update an existing PPTX binary file in Google Drive in-place."""
    from googleapiclient.http import MediaFileUpload

    media = MediaFileUpload(str(pptx_path), mimetype=PPTX_MIME, resumable=True)
    updated = (
        drive_service.files()
        .update(
            fileId=file_id,
            media_body=media,
            fields="id, name, mimeType, webViewLink",
        )
        .execute()
    )
    return updated


def sync_presentation(
    pptx_path: Path,
    title: str,
    file_id: str | None = None,
    folder_id: str | None = None,
    credentials_path: str | None = None,
    as_gslides: bool = True,
):
    creds = get_credentials(credentials_path)
    drive_service = build_drive_service(creds)

    # Mode 1: Target File ID is explicitly provided
    if file_id:
        print(f"Checking target file ID: {file_id}...")
        try:
            target_file = (
                drive_service.files()
                .get(fileId=file_id, fields="id, name, mimeType, parents, webViewLink")
                .execute()
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch file '{file_id}'. Ensure the file exists and is shared with your Service Account email."
            ) from e

        mime = target_file.get("mimeType", "")
        parents = target_file.get("parents", [])
        parent_folder = parents[0] if parents else None

        if mime == PPTX_MIME:
            print(f"Updating existing PPTX file '{target_file.get('name')}' in-place...")
            res = update_pptx_file(drive_service, file_id, pptx_path)
            link = res.get("webViewLink")
            print(f"Updated successfully! Link: {link}")
            return res

        elif mime == GSLIDES_MIME:
            print(
                f"Target '{target_file.get('name')}' is a native Google Slides presentation."
            )
            print("Creating updated native Google Slides presentation...")
            res = upload_new_presentation(
                drive_service,
                pptx_path,
                title=title or target_file.get("name", "Presentation"),
                folder_id=parent_folder or folder_id,
                as_gslides=True,
            )
            # Trash the old presentation if in the same folder so only the new one remains
            if parent_folder:
                try:
                    drive_service.files().update(
                        fileId=file_id, body={"trashed": True}
                    ).execute()
                    print(f"Archived previous version ({file_id}) to trash.")
                except Exception as trash_err:
                    print(f"Note: Could not trash previous version: {trash_err}")

            link = res.get("webViewLink")
            print(f"Sync complete! Link: {link}")
            return res
        else:
            print(f"Target file has MIME type '{mime}'. Updating binary content...")
            res = update_pptx_file(drive_service, file_id, pptx_path)
            link = res.get("webViewLink")
            print(f"Updated successfully! Link: {link}")
            return res

    # Mode 2: Folder ID is provided (sync or create within folder)
    if folder_id:
        target_title = title or pptx_path.stem
        print(f"Searching folder '{folder_id}' for existing presentation named '{target_title}'...")
        query = (
            f"'{folder_id}' in parents and trashed = false and "
            f"(name = '{target_title}' or name = '{target_title}.pptx')"
        )
        results = (
            drive_service.files()
            .list(q=query, fields="files(id, name, mimeType, webViewLink)")
            .execute()
        )
        existing_files = results.get("files", [])

        if existing_files:
            existing = existing_files[0]
            print(f"Found existing item '{existing['name']}' (ID: {existing['id']}).")
            if existing["mimeType"] == PPTX_MIME:
                res = update_pptx_file(drive_service, existing["id"], pptx_path)
                print(f"Updated in-place! Link: {res.get('webViewLink')}")
                return res
            else:
                # Native Google Slides in folder: upload new and trash old
                res = upload_new_presentation(
                    drive_service,
                    pptx_path,
                    title=target_title,
                    folder_id=folder_id,
                    as_gslides=as_gslides,
                )
                try:
                    drive_service.files().update(
                        fileId=existing["id"], body={"trashed": True}
                    ).execute()
                    print(f"Replaced previous presentation {existing['id']}.")
                except Exception as trash_err:
                    print(f"Note: Could not trash previous version: {trash_err}")
                print(f"Sync complete! Link: {res.get('webViewLink')}")
                return res
        else:
            print(f"No existing presentation found. Uploading fresh presentation to folder...")
            res = upload_new_presentation(
                drive_service,
                pptx_path,
                title=target_title,
                folder_id=folder_id,
                as_gslides=as_gslides,
            )
            print(f"Upload complete! Link: {res.get('webViewLink')}")
            return res

    # Mode 3: No file ID and no folder ID specified -> upload to root of Drive
    target_title = title or pptx_path.stem
    print("No folder or file ID specified. Uploading directly to Google Drive...")
    res = upload_new_presentation(
        drive_service,
        pptx_path,
        title=target_title,
        as_gslides=as_gslides,
    )
    print(f"Upload complete! Link: {res.get('webViewLink')}")
    return res


def main():
    parser = argparse.ArgumentParser(
        description="Upload / sync presentation to Google Drive & Google Slides"
    )
    parser.add_argument("pptx_path", type=Path, help="Path to .pptx presentation file")
    parser.add_argument(
        "--title",
        type=str,
        default="Which Exposure Matters - Presentation",
        help="Title of the presentation on Google Slides",
    )
    parser.add_argument(
        "--file-id",
        type=str,
        default=os.environ.get("GDRIVE_PRESENTATION_ID")
        or os.environ.get("GDRIVE_FILE_ID"),
        help="Google Drive File / Presentation ID to update",
    )
    parser.add_argument(
        "--folder-id",
        type=str,
        default=os.environ.get("GDRIVE_FOLDER_ID"),
        help="Google Drive Folder ID where presentation should be placed / updated",
    )
    parser.add_argument(
        "--credentials",
        type=str,
        default=None,
        help="Path to GCP Service Account JSON key file",
    )
    parser.add_argument(
        "--no-convert",
        action="store_true",
        help="Upload as raw PPTX without converting to native Google Slides",
    )

    args = parser.parse_args()

    pptx_path = args.pptx_path.resolve()
    if not pptx_path.is_file():
        print(f"Error: PPTX file '{pptx_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    sync_presentation(
        pptx_path=pptx_path,
        title=args.title,
        file_id=args.file_id,
        folder_id=args.folder_id,
        credentials_path=args.credentials,
        as_gslides=not args.no_convert,
    )


if __name__ == "__main__":
    main()
