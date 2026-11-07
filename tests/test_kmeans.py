
import unittest
import numpy as np

from kmeans import kmeans


class TestKMeans(unittest.TestCase):

    def setUp(self):
        self.data = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [10.0, 10.0],
            [10.0, 11.0],
            [11.0, 10.0]
        ])

    def test_output_shapes(self):
        labels, centroids = kmeans(
            self.data, k=2, random_state=42
        )

        self.assertEqual(labels.shape, (6,))
        self.assertEqual(centroids.shape, (2, 2))

    def test_cluster_count(self):
        labels, centroids = kmeans(
            self.data, k=2, random_state=42
        )

        self.assertEqual(len(np.unique(labels)), 2)

    def test_reproducibility(self):
        labels_1, centroids_1 = kmeans(
            self.data, k=2, random_state=42
        )

        labels_2, centroids_2 = kmeans(
            self.data, k=2, random_state=42
        )

        np.testing.assert_array_equal(labels_1, labels_2)

        np.testing.assert_allclose(
            centroids_1, centroids_2
        )

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            kmeans(self.data, k=0)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            kmeans(np.array([1, 2, 3]), k=2)


if __name__ == "__main__":
    unittest.main()
