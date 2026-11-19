import unittest
from exposure.spatial_sector import sector_holds,exists_sector,simulate
from exposure.rules import SceneObject

class SectorTests(unittest.TestCase):
    def test_cardinal_directions_and_distance(self):
        c=SceneObject('red','circle',30,30)
        left=SceneObject('blue','square',10,30)
        diagonal=SceneObject('green','triangle',10,10)
        self.assertTrue(sector_holds(left,c,direction='left'))
        self.assertTrue(sector_holds(c,left,direction='right'))
        self.assertFalse(sector_holds(diagonal,c,direction='left'))
        self.assertFalse(sector_holds(left,c,direction='left',max_distance=19))
        self.assertFalse(sector_holds(c,c,direction='left'))

    def test_binding_and_distinct_roles(self):
        a=SceneObject('red','circle',10,30);b=SceneObject('blue','square',30,30)
        self.assertTrue(exists_sector((a,b),direction='left',subject_filter={'color':'red'},target_filter={'shape':'square'}))
        self.assertFalse(exists_sector((a,b),direction='left',subject_filter={'color':'blue'},target_filter={'shape':'circle'}))
        self.assertFalse(exists_sector((a,),direction='left'))
        with self.assertRaises(ValueError):exists_sector((a,b),direction='left',subject_filter={'x':10})

    def test_determinism_and_validation(self):
        self.assertEqual(simulate(100,1),simulate(100,1))
        with self.assertRaises(ValueError): simulate(0,1)
        with self.assertRaises(ValueError): sector_holds(SceneObject('red','circle',1,1),SceneObject('blue','square',20,1),direction='left',half_angle_degrees=45)

if __name__=='__main__':unittest.main()
