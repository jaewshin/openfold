import unittest

from align.embed_align import gate_and_rank, smith_waterman_affine, build_alignment_result
from pipeline_a.embeddings import build_embedder


class TestPipelineSWAlignment(unittest.TestCase):
    def setUp(self):
        self.embedder = build_embedder("aa_onehot")

    def test_perfect_match_alignment(self):
        query = "ACDE"
        candidate = "ACDE"
        trace = smith_waterman_affine(
            self.embedder.embed_residues(query),
            self.embedder.embed_residues(candidate),
            scale=5.0,
            bias=-1.0,
            gap_open=-2.0,
            gap_extend=-0.5,
        )
        self.assertGreater(trace.score, 0.0)
        self.assertEqual(trace.q_start, 0)
        self.assertEqual(trace.s_start, 0)
        self.assertEqual(trace.ops, ["M", "M", "M", "M"])

    def test_local_motif_alignment(self):
        query = "ACDEFGH"
        candidate = "XXCDEYY"
        trace = smith_waterman_affine(
            self.embedder.embed_residues(query),
            self.embedder.embed_residues(candidate),
            scale=5.0,
            bias=-1.0,
            gap_open=-2.0,
            gap_extend=-0.5,
        )
        self.assertGreater(trace.score, 0.0)
        self.assertEqual(trace.q_start, 1)
        self.assertEqual(trace.s_start, 2)
        self.assertEqual(trace.ops, ["M", "M", "M"])

    def test_gate_and_rank(self):
        query = "ACDEFGHIK"
        candidate = "ACDEFGHIK"
        trace = smith_waterman_affine(
            self.embedder.embed_residues(query),
            self.embedder.embed_residues(candidate),
            scale=5.0,
            bias=-1.0,
            gap_open=-2.0,
            gap_extend=-0.5,
        )
        row = build_alignment_result(
            seq_id="seq1",
            ann_score=1.0,
            sequence=candidate,
            trace=trace,
            query_length=len(query),
        )

        kept = gate_and_rank(
            [row],
            min_query_coverage=0.8,
            min_aligned_query_len=5,
            min_score_density=0.1,
            max_gap_frac=0.5,
            max_rows=1,
        )
        self.assertEqual(len(kept), 1)


if __name__ == "__main__":
    unittest.main()
