import unittest

from scripts.upload_to_gslides import replacement_requests


class GoogleSlidesReplacementTests(unittest.TestCase):
    def test_replaces_every_old_slide_after_creating_new_slides(self):
        old = [{'objectId': 'old_1'}, {'objectId': 'old_2'}, {'objectId': 'old_3'}]
        requests = replacement_requests(old, ['https://example.com/1.png', 'https://example.com/2.png'], run_id='abc')
        self.assertEqual([next(iter(item)) for item in requests], [
            'createSlide', 'updatePageProperties', 'createSlide',
            'updatePageProperties', 'deleteObject', 'deleteObject', 'deleteObject'])
        self.assertEqual(requests[0]['createSlide']['objectId'], 'sync_abc_000')
        self.assertEqual(requests[1]['updatePageProperties']['objectId'], 'sync_abc_000')
        self.assertEqual(requests[3]['updatePageProperties']['pageProperties']
                         ['pageBackgroundFill']['stretchedPictureFill']['contentUrl'],
                         'https://example.com/2.png')
        self.assertEqual([r['deleteObject']['objectId'] for r in requests[4:]],
                         ['old_1', 'old_2', 'old_3'])

    def test_empty_input_does_not_delete_the_existing_deck(self):
        with self.assertRaises(ValueError):
            replacement_requests([{'objectId': 'old'}], [])


if __name__ == '__main__':
    unittest.main()
