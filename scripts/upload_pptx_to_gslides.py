#!/usr/bin/env python3
"""Replace an existing Google Slides presentation with an editable PPTX import."""
import argparse
import os
from pathlib import Path
from urllib.request import urlopen
import xml.etree.ElementTree as ET
from zipfile import BadZipFile, ZipFile

from scripts.upload_to_gslides import get_credentials

PPTX_MIME = 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
SLIDES_MIME = 'application/vnd.google-apps.presentation'


def slide_count(path: Path) -> int:
    """Reject missing, corrupt, or empty presentations before touching the target."""
    try:
        with ZipFile(path) as archive:
            root = ET.fromstring(archive.read('ppt/presentation.xml'))
    except (OSError, BadZipFile, KeyError, ET.ParseError) as exc:
        raise ValueError(f'Invalid PowerPoint file: {path}') from exc
    return len(root.findall('.//{http://schemas.openxmlformats.org/presentationml/2006/main}sldId'))


def update_presentation(drive, media, presentation_id: str, expected_slides: int):
    """Update the existing Google file's contents while retaining its ID and sharing."""
    target = drive.files().get(fileId=presentation_id, fields='id,name,mimeType',
                               supportsAllDrives=True).execute()
    if target.get('mimeType') != SLIDES_MIME:
        raise ValueError('Target must be an existing Google Slides presentation')
    result = drive.files().update(fileId=presentation_id, media_body=media,
                                  fields='id,name,mimeType,webViewLink',
                                  supportsAllDrives=True).execute()
    if result.get('id') != presentation_id or result.get('mimeType') != SLIDES_MIME:
        raise RuntimeError('Drive returned an unexpected updated file')
    print(f'Updated Google Slides: {result.get("name", "presentation")} ({expected_slides} source slides)')
    print(result.get('webViewLink') or f'https://docs.google.com/presentation/d/{presentation_id}/edit')


def verify_presentation(slides_service, presentation_id: str, expected_slides: int):
    """Verify the conversion kept the page count and some editable text."""
    presentation = slides_service.presentations().get(presentationId=presentation_id).execute()
    slides = presentation.get('slides', [])
    text_shapes = sum(
        1 for slide in slides for element in slide.get('pageElements', [])
        if any(item.get('textRun', {}).get('content', '').strip()
               for item in element.get('shape', {}).get('text', {}).get('textElements', []))
    )
    if len(slides) != expected_slides or not text_shapes:
        raise RuntimeError(f'Converted deck has {len(slides)} slides and {text_shapes} text shapes; '
                           f'expected {expected_slides} slides and editable text')
    # The methodology slide's two co-occurrence tables are essential evidence.
    if expected_slides == 12:
        tables = [element['table'] for element in slides[3].get('pageElements', [])
                  if 'table' in element]
        if len(tables) < 2:
            raise RuntimeError(f'Converted slide 4 has {len(tables)} tables; expected two')
        print(f'Verified {len(tables)} editable comparison tables on slide 4')
        if not any('image' in element for element in slides[4].get('pageElements', [])):
            raise RuntimeError('Converted slide 5 is missing the LaTeX equation image')
        print('Verified the LaTeX equation image on slide 5')
        cover_images = [element['image'] for element in slides[0].get('pageElements', [])
                        if 'image' in element]
        if len(cover_images) < 3:
            raise RuntimeError(f'Converted title slide has {len(cover_images)} images; expected three')
        print(f'Verified {len(cover_images)} diffusion illustrations on the title slide')
        images = [element['image'] for element in slides[8].get('pageElements', [])
                  if 'image' in element]
        if len(images) < 2:
            raise RuntimeError(f'Converted slide 9 has {len(images)} images; expected two')
        print(f'Verified {len(images)} qualitative images on slide 9')
    print(f'Verified {len(slides)} slides and {text_shapes} editable text shapes')
    return slides


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pptx', type=Path, required=True)
    parser.add_argument('--presentation-id', default=os.environ.get('GDRIVE_PRESENTATION_ID'))
    parser.add_argument('--credentials')
    parser.add_argument('--expected-slides', type=int, default=12)
    parser.add_argument('--slide1-thumbnail', type=Path,
                        help='Save the introduction thumbnail for layout inspection')
    parser.add_argument('--slide3-thumbnail', type=Path,
                        help='Save a Google-rendered thumbnail for layout inspection')
    parser.add_argument('--slide4-thumbnail', type=Path,
                        help='Save the methodology-slide thumbnail for layout inspection')
    parser.add_argument('--slide5-thumbnail', type=Path,
                        help='Save the architecture-slide thumbnail for layout inspection')
    parser.add_argument('--slide9-thumbnail', type=Path,
                        help='Save the hypothesis-slide thumbnail for layout inspection')
    args = parser.parse_args()
    if not args.presentation_id:
        parser.error('Pass --presentation-id or set GDRIVE_PRESENTATION_ID')
    try:
        count = slide_count(args.pptx)
    except ValueError as exc:
        parser.error(str(exc))
    if count != args.expected_slides or count == 0:
        parser.error(f'Expected {args.expected_slides} slides, found {count}')
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    credentials = get_credentials(args.credentials)
    drive = build('drive', 'v3', credentials=credentials)
    media = MediaFileUpload(str(args.pptx), mimetype=PPTX_MIME, resumable=False)
    update_presentation(drive, media, args.presentation_id, count)
    slides_service = build('slides', 'v1', credentials=credentials)
    slides = verify_presentation(slides_service, args.presentation_id, count)
    for index, output in ((0, args.slide1_thumbnail), (2, args.slide3_thumbnail), (3, args.slide4_thumbnail),
                          (4, args.slide5_thumbnail),
                          (8, args.slide9_thumbnail)):
        if not output:
            continue
        thumbnail = slides_service.presentations().pages().getThumbnail(
            presentationId=args.presentation_id, pageObjectId=slides[index]['objectId'],
            thumbnailProperties_mimeType='PNG', thumbnailProperties_thumbnailSize='LARGE',
        ).execute()
        with urlopen(thumbnail['contentUrl'], timeout=30) as response:
            image = response.read()
        if not image.startswith(b'\x89PNG\r\n\x1a\n'):
            raise RuntimeError('Google Slides returned an invalid thumbnail')
        output.write_bytes(image)


if __name__ == '__main__':
    main()
