import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from zipfile import ZipFile

from scripts.upload_pptx_to_gslides import SLIDES_MIME, slide_count, update_presentation, verify_presentation


class PowerPointImportTests(unittest.TestCase):
    def test_counts_slides_and_rejects_invalid_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'presentation.pptx'
            with ZipFile(path, 'w') as archive:
                archive.writestr('ppt/presentation.xml', '<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:sldIdLst><p:sldId/><p:sldId/></p:sldIdLst></p:presentation>')
            self.assertEqual(slide_count(path), 2)
            with self.assertRaises(ValueError):
                slide_count(path.parent / 'missing.pptx')

    def test_updates_only_an_existing_slides_file(self):
        drive = MagicMock()
        drive.files().get().execute.return_value = {'id': 'deck', 'mimeType': SLIDES_MIME}
        drive.files().update().execute.return_value = {'id': 'deck', 'mimeType': SLIDES_MIME}
        media = object()
        update_presentation(drive, media, 'deck', 12)
        drive.files().update.assert_called_with(fileId='deck', media_body=media,
            fields='id,name,mimeType,webViewLink', supportsAllDrives=True)
        drive.reset_mock()
        drive.files().get().execute.return_value = {'id': 'deck', 'mimeType': 'application/pdf'}
        with self.assertRaises(ValueError):
            update_presentation(drive, media, 'deck', 12)
        drive.files().update.assert_not_called()

    def test_verifies_editable_text_and_page_count(self):
        service = MagicMock()
        slide = {'pageElements': [{'shape': {'text': {'textElements': [
            {'textRun': {'content': 'Editable text'}}]}}}]}
        service.presentations().get().execute.return_value = {'slides': [slide] * 12}
        verify_presentation(service, 'deck', 12)
        service.presentations().get().execute.return_value = {'slides': [{}] * 12}
        with self.assertRaisesRegex(RuntimeError, 'editable text'):
            verify_presentation(service, 'deck', 12)


if __name__ == '__main__':
    unittest.main()
