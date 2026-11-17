#!/usr/bin/env python3
"""Sync paper files directly to an Overleaf project using an Overleaf session cookie.

Works with Overleaf Free tier (no Git integration or paid plan required).
"""

import argparse
import io
import os
import sys
from pathlib import Path

# Files/extensions to ignore during upload
IGNORE_EXTENSIONS = {
    ".aux",
    ".log",
    ".out",
    ".toc",
    ".bbl",
    ".blg",
    ".synctex.gz",
    ".fdb_latexmk",
    ".fls",
    ".pyc",
}
IGNORE_DIRS = {".git", "__pycache__", ".venv"}


def sync_paper(
    project_id: str,
    cookie_value: str,
    paper_dir: Path,
    server_url: str = "https://www.overleaf.com",
):
    try:
        from olsync.olclient import OverleafClient
    except ImportError:
        raise RuntimeError(
            "Missing 'overleaf-sync' package. Run: pip install overleaf-sync"
        )

    cookie_value = cookie_value.strip()
    cookies = {"overleaf_session2": cookie_value}

    print(f"Authenticating with Overleaf ({server_url})...")
    temp_client = OverleafClient(cookies, base_url=server_url)
    csrf = temp_client.fetch_csrf()
    if not csrf:
        print(
            "\nERROR: Authentication failed! The provided overleaf_session2 cookie is invalid or expired.\n"
            "Please check that you copied the complete cookie value from your logged-in browser.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Authenticated successfully!")
    # Initialize client with csrf token so uploads and folder creation are authorized
    client = OverleafClient(cookies, csrf=csrf, base_url=server_url)
    print(f"Fetching Overleaf project details for ID: {project_id}...")
    try:
        project_infos = client.get_project_infos(project_id)
    except Exception as e:
        print(
            f"\nERROR: Could not fetch project '{project_id}': {e}\n"
            "Ensure the project ID is correct and that your Overleaf account has opened/joined this project.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    project_name = project_infos.get("name", project_id)
    print(f"Connected to project: '{project_name}'")

    # Gather local files to upload
    files_to_upload = []
    for root, dirs, files in os.walk(paper_dir):
        # Filter out ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for f in files:
            file_path = Path(root) / f
            if file_path.suffix.lower() in IGNORE_EXTENSIONS:
                continue
            if f.startswith("."):
                continue
            rel_path = file_path.relative_to(paper_dir)
            files_to_upload.append((file_path, str(rel_path)))

    print(f"\nUploading {len(files_to_upload)} file(s) from '{paper_dir}' to Overleaf:")
    uploaded = 0
    failed = 0

    for file_path, rel_name in sorted(files_to_upload, key=lambda x: x[1]):
        with open(file_path, "rb") as fp:
            content = fp.read()

        size = len(content)
        print(f"  -> Uploading: {rel_name} ({size:,} bytes)...", end="", flush=True)
        try:
            success = client.upload_file(
                project_id,
                project_infos,
                rel_name,
                size,
                io.BytesIO(content),
            )
            if success:
                print(" OK")
                uploaded += 1
            else:
                print(" FAILED (rejected by Overleaf)")
                failed += 1
        except Exception as upload_err:
            print(f" ERROR: {upload_err}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"Sync complete: {uploaded} uploaded, {failed} failed.")
    print(f"View in Overleaf: https://www.overleaf.com/project/{project_id}")
    print("=" * 50 + "\n")
    if failed > 0:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Sync paper files directly to Overleaf without paid Git"
    )
    parser.add_argument(
        "--project-id",
        type=str,
        default=os.environ.get("OVERLEAF_PROJECT_ID") or "6afc917c5676c2467e5ab0f2",
        help="Overleaf Project ID (default: 6afc917c5676c2467e5ab0f2 or OVERLEAF_PROJECT_ID env var)",
    )
    parser.add_argument(
        "--cookie",
        type=str,
        default=os.environ.get("OVERLEAF_COOKIE"),
        help="Value of overleaf_session2 cookie (or set OVERLEAF_COOKIE env var)",
    )
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=Path("paper"),
        help="Local directory containing paper files (default: paper)",
    )
    parser.add_argument(
        "--server",
        type=str,
        default="https://www.overleaf.com",
        help="Overleaf server URL (default: https://www.overleaf.com)",
    )

    args = parser.parse_args()

    if not args.cookie:
        print(
            "Error: Overleaf session cookie is required.\n"
            "Pass --cookie or set the OVERLEAF_COOKIE environment variable.\n\n"
            "To get your cookie:\n"
            "1. Open overleaf.com in your browser (logged in).\n"
            "2. Open Developer Tools (F12 or Right Click -> Inspect) -> Application tab (Storage tab in Firefox).\n"
            "3. Under Cookies -> https://www.overleaf.com, find 'overleaf_session2' and copy its value.",
            file=sys.stderr,
        )
        sys.exit(1)

    paper_dir = args.paper_dir.resolve()
    if not paper_dir.is_dir():
        print(f"Error: Paper directory '{paper_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    sync_paper(
        project_id=args.project_id,
        cookie_value=args.cookie,
        paper_dir=paper_dir,
        server_url=args.server,
    )


if __name__ == "__main__":
    main()
