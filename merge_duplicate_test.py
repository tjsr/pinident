import unittest
from boxdata import BoxData, Coordinate
from scrubberframe import merge_duplicate_boxes

class TestMergeDuplicateBoxes(unittest.TestCase):
    def test_merge_boxes_with_overlapping_tags(self):
        coord = (10, 20, 30, 40)
        box1 = BoxData(coords=coord, tags=['cat', 'dog'], source='user')
        box2 = BoxData(coords=coord, tags=['dog', 'mouse'], source='user')
        boxes = [box1, box2]
        merged = merge_duplicate_boxes(boxes)
        self.assertEqual(len(merged), 1)
        merged_tags = set(merged[0].tags)
        self.assertSetEqual(merged_tags, {'cat', 'dog', 'mouse'})
        self.assertEqual(merged[0].coords, coord)
        # 'dog' should only appear once
        self.assertEqual(merged[0].tags.count('dog'), 1)

    def test_no_merge_for_different_coords(self):
        box1 = BoxData(coords=(1, 2, 3, 4), tags=['a'], source='user')
        box2 = BoxData(coords=(5, 6, 7, 8), tags=['b'], source='user')
        boxes = [box1, box2]
        merged = merge_duplicate_boxes(boxes)
        self.assertEqual(len(merged), 2)
        coords = {box.coords for box in merged}
        self.assertSetEqual(coords, {(1, 2, 3, 4), (5, 6, 7, 8)})

if __name__ == '__main__':
    unittest.main()