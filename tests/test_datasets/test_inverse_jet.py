import unittest

import cv2
import numpy as np

from projects.vma.tools.recover_sd_intensity import InverseJet


class TestInverseJet(unittest.TestCase):

    def test_exact_palette_and_arbitrary_colors(self):
        inverse = InverseJet()
        ramp = np.arange(256, dtype=np.uint8).reshape(16, 16)
        np.testing.assert_array_equal(
            inverse(cv2.applyColorMap(ramp, cv2.COLORMAP_JET)), ramp)
        rng = np.random.default_rng(17)
        colors = rng.integers(0, 256, (16, 32, 3), dtype=np.uint8)
        delta = colors.reshape(-1, 1, 3).astype(
            np.int32) - inverse.palette[None]
        expected = (delta * delta).sum(axis=2).argmin(axis=1).reshape(16, 32)
        result = inverse(colors)
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(inverse(colors), result)
        np.testing.assert_array_equal(inverse(colors[::-1]), result[::-1])
        self.assertEqual(result.dtype, np.uint8)
        with self.assertRaises(ValueError):
            inverse(colors.astype(float))


if __name__ == '__main__':
    unittest.main()
