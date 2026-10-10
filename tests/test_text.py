import unittest

import numpy as np

from nlp_assignment.text import mean_document_vector, tokenize, vectorize_documents


class TextUtilitiesTest(unittest.TestCase):
    def test_tokenize_normalizes_english_text(self):
        self.assertEqual(
            tokenize("Apple's NEW model costs 100 dollars."),
            ["apple's", "new", "model", "costs", "100", "dollars"],
        )

    def test_mean_document_vector_ignores_oov(self):
        vectors = {
            "apple": np.array([1.0, 0.0], dtype=np.float32),
            "banana": np.array([0.0, 1.0], dtype=np.float32),
        }
        vector, known, total = mean_document_vector(
            ["apple", "missing", "banana"], vectors, 2
        )
        np.testing.assert_allclose(vector, [0.5, 0.5])
        self.assertEqual((known, total), (2, 3))

    def test_vectorize_empty_document_returns_zero(self):
        matrix, stats = vectorize_documents([[], ["missing"]], {}, 3)
        np.testing.assert_array_equal(matrix, np.zeros((2, 3), dtype=np.float32))
        self.assertEqual(stats["zero_vector_documents"], 2)


if __name__ == "__main__":
    unittest.main()

