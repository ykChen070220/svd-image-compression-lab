import unittest
from io import BytesIO
import numpy as np
from PIL import Image
from core import (decompose, reconstruct, display_pixels, metrics, storage,
                  theoretical_curve, prepare_image, demo_image, factor_bytes)


class CompressionTests(unittest.TestCase):
    def test_full_rank_rgb_and_gray(self):
        rng = np.random.default_rng(42)
        for shape in [(12, 9), (12, 9, 3), (1, 6, 3)]:
            pixels = rng.integers(0, 256, shape, dtype=np.uint8)
            factors = decompose(pixels)
            result = reconstruct(factors, min(shape[:2]), len(shape) == 2)
            np.testing.assert_allclose(result, pixels, atol=1e-10)
            np.testing.assert_array_equal(display_pixels(result), pixels)
            self.assertEqual(metrics(pixels, display_pixels(result))[0], 0)

    def test_tail_energy_matches_actual_error(self):
        pixels = np.random.default_rng(2).integers(0, 256, (10, 8, 3), dtype=np.uint8)
        factors = decompose(pixels)
        curve = theoretical_curve(factors, pixels.shape)
        previous = float("inf")
        for row in curve:
            a = reconstruct(factors, row["k"])
            mse = np.mean((pixels.astype(float)-a)**2)
            self.assertAlmostEqual(mse, row["理论MSE"], places=7)
            self.assertLessEqual(mse, previous + 1e-10)
            previous = mse

    def test_zero_and_storage_and_export(self):
        pixels = np.zeros((5, 7), dtype=np.uint8)
        factors = decompose(pixels)
        self.assertEqual(theoretical_curve(factors, pixels.shape)[0]["保留能量(%)"], 100)
        self.assertEqual(storage((5, 7, 3), 2), (105, 78, 312))
        for k in [0, 3, 5]:
            with np.load(BytesIO(factor_bytes(factors, k, pixels.shape))) as archive:
                self.assertEqual(sum(archive[n].nbytes for n in ["U_0", "s_0", "Vh_0"]),
                                 storage(pixels.shape, k)[2])

    def test_input_handling(self):
        pixels, size = prepare_image(demo_image(), 128, True)
        self.assertEqual(size, (384, 256))
        self.assertEqual(pixels.ndim, 2)
        self.assertEqual(max(pixels.shape), 128)
        stream = BytesIO()
        Image.new("RGBA", (2, 2), (0, 0, 0, 0)).save(stream, format="PNG")
        pixels, _ = prepare_image(stream.getvalue())
        self.assertTrue(np.all(pixels == 255))
        with self.assertRaises(OSError):
            prepare_image(b"not an image")

    def test_export_round_trip(self):
        pixels = np.random.default_rng(7).integers(0, 256, (9, 7, 3), dtype=np.uint8)
        factors = decompose(pixels)
        for dtype in ["float32", "float64"]:
            with np.load(BytesIO(factor_bytes(factors, 3, pixels.shape, dtype))) as archive:
                restored = np.stack([(archive[f"U_{c}"] * archive[f"s_{c}"]) @ archive[f"Vh_{c}"]
                                     for c in range(3)], axis=-1)
                np.testing.assert_allclose(restored, reconstruct(factors, 3), atol=1e-4)


if __name__ == "__main__":
    unittest.main()
