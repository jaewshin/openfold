import tempfile
import unittest
from pathlib import Path

from msa.a3m_writer import (
    project_trace_to_a3m_row,
    validate_a3m_invariants,
    write_a3m,
)


class TestPipelineA3MWriter(unittest.TestCase):
    def test_insertion_and_deletion_projection(self):
        row = project_trace_to_a3m_row(
            query_length=6,
            candidate_sequence="ACXDFG",
            q_start=0,
            s_start=0,
            ops=["M", "M", "I", "M", "D", "M", "M"],
            seq_id="seq1",
            header="retriever|seq1",
        )
        self.assertEqual(row.sequence, "ACxD-FG")

    def test_partial_local_alignment_projection(self):
        row = project_trace_to_a3m_row(
            query_length=8,
            candidate_sequence="CDEF",
            q_start=2,
            s_start=0,
            ops=["M", "M", "M", "M"],
            seq_id="seq2",
            header="retriever|seq2",
        )
        self.assertEqual(row.sequence, "--CDEF--")

    def test_openfold_a3m_invariants(self):
        query = "ACDEFG"
        row = project_trace_to_a3m_row(
            query_length=len(query),
            candidate_sequence="ACXDFG",
            q_start=0,
            s_start=0,
            ops=["M", "M", "I", "M", "D", "M", "M"],
            seq_id="seq1",
            header="retriever|seq1",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.a3m"
            write_a3m(
                output_path=path,
                query_id="query",
                query_sequence=query,
                rows=[row],
            )

            stats = validate_a3m_invariants(a3m_path=path, query_sequence=query)
            self.assertEqual(stats["query_length"], len(query))
            self.assertEqual(stats["num_sequences"], 2)


if __name__ == "__main__":
    unittest.main()
