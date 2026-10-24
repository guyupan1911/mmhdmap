import unittest

import numpy as np
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon

from mmhdmap.visualization.map_projection import project_map_geometries


class TestMapProjection(unittest.TestCase):

    def setUp(self):
        # Ego X-forward/Y-left/Z-up -> camera X-right/Y-down/Z-forward.
        self.ego2cam = np.array(
            [[0, -1, 0, 0], [0, 0, -1, 1.5], [1, 0, 0, 0], [0, 0, 0, 1]],
            dtype=float)
        self.intrinsic = np.array([[50, 0, 50], [0, 50, 50], [0, 0, 1]])

    def project(self, geometries, **kwargs):
        options = dict(
            local2global=np.eye(3),
            global2cam=self.ego2cam,
            cam2img=self.intrinsic,
            image_shape=(101, 101),
            ground_z=0)
        options.update(kwargs)
        return project_map_geometries(geometries, **options)

    def test_known_pixels_and_ground_offset(self):
        geometries = dict(lane_divider=[LineString([(5, -2), (5, 2)])])
        pixels = self.project(geometries)['lane_divider']
        np.testing.assert_allclose(pixels, [[[70, 65], [30, 65]]])
        raised = self.project(geometries, ground_z=0.5)['lane_divider']
        np.testing.assert_allclose(raised, [[[70, 60], [30, 60]]])

    def test_full_pose_and_planar_map_frame(self):
        yaw, pitch = 0.7, 0.2
        c, s = np.cos(yaw), np.sin(yaw)
        planar = np.array([[c, -s, 20], [s, c, -10], [0, 0, 1]])
        yaw_rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        c, s = np.cos(pitch), np.sin(pitch)
        pitch_rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        pose = np.eye(4)
        pose[:3, :3] = yaw_rotation @ pitch_rotation
        pose[:3, 3] = [20, -10, 3]
        xy = np.array([[5, -2], [5, 2]])
        pixels = self.project(
            dict(lane_divider=[LineString(xy)]),
            local2global=planar,
            global2cam=self.ego2cam @ np.linalg.inv(pose),
            ground_z=3)['lane_divider'][0]
        # Independently transform each point through global, ego and camera.
        expected = []
        for x, y in xy:
            global_xy = planar @ [x, y, 1]
            global_point = np.array([*global_xy[:2], 3, 1])
            ego_point = np.linalg.solve(pose, global_point)
            camera_point = self.ego2cam @ ego_point
            pixel = self.intrinsic @ camera_point[:3]
            expected.append(pixel[:2] / pixel[2])
        np.testing.assert_allclose(pixels, expected)
        self.assertFalse(np.allclose(pixels[:, 1], 65))

    def test_near_and_image_clipping_retains_crossing_segments(self):
        geometries = dict(lane_divider=[
            LineString([(-2, 0), (-1, 0)]),
            LineString([(-1, 0), (2, 0)]),
            LineString([(5, -20), (5, 20)]),
        ])
        pixels = self.project(geometries)['lane_divider']
        np.testing.assert_allclose(
            pixels, [[[50, 100], [50, 87.5]], [[100, 65], [0, 65]]])
        self.assertTrue(np.isfinite(pixels).all())
        # A higher near plane clips a segment before the image boundaries do.
        line = dict(lane_divider=[LineString([(1, 0), (4, 0)])])
        pixels = self.project(line, near_clip=2)['lane_divider']
        np.testing.assert_allclose(pixels, [[[50, 87.5], [50, 68.75]]])

    def test_parts_holes_and_empty_layers(self):
        lines = MultiLineString([[(5, -3), (5, -2)], [(5, 2), (5, 3)]])
        polygon = Polygon([(5, -2), (10, -2), (10, 2), (5, 2)],
                          holes=[[(6, -1), (9, -1), (9, 1), (6, 1)]])
        result = self.project(
            dict(
                lane_divider=[lines],
                ped_crossing=[MultiPolygon([polygon])],
                road_divider=[]))
        self.assertEqual(result['lane_divider'].shape, (2, 2, 2))
        self.assertEqual(result['ped_crossing'].shape, (8, 2, 2))
        self.assertEqual(result['road_divider'].shape, (0, 2, 2))
        self.assertEqual(len(polygon.interiors), 1)

    def test_invalid_projection_parameters(self):
        for options, message in [
            (dict(near_clip=0), 'near_clip'),
            (dict(ground_z=np.nan), 'ground_z'),
            (dict(local2global=np.eye(4)), 'local2global'),
            (dict(cam2img=np.zeros((3, 3))), 'last row'),
            (dict(image_shape=(0, 100)), 'image_shape'),
        ]:
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, message):
                    self.project({}, **options)


if __name__ == '__main__':
    unittest.main()
