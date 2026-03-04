import argparse
import tempfile
import unittest
from pathlib import Path

from db.build_db_embeddings import build_db_embeddings
from db.build_faiss_index import build_faiss_index, build_ids_offsets
from pipeline_a_openfold import run_pipeline


def _faiss_available() -> bool:
    try:
        import faiss  # noqa: F401

        return True
    except Exception:
        return False


def _openfold_datapipeline_available() -> bool:
    try:
        from openfold.data.data_pipeline import DataPipeline  # noqa: F401

        return True
    except Exception:
        return False


@unittest.skipUnless(
    _faiss_available() and _openfold_datapipeline_available(),
    "Requires FAISS and OpenFold data pipeline dependencies",
)
class TestPipelineAEndToEnd(unittest.TestCase):
    def test_full_pipeline_to_openfold_feature_ingestion(self):
        query_seq = "ACDEFGHIK"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            db_fasta = tmp / "db.fasta"
            db_fasta.write_text(
                "\n".join(
                    [
                        ">seqA",
                        "ACDEFGHIK",
                        ">seqB",
                        "ACDEFGHIM",
                        ">seqC",
                        "TTTTTTTTT",
                        ">seqD",
                        "CDEFG",
                        ">seqE",
                        "ACDEYGHIK",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            query_fasta = tmp / "query.fasta"
            query_fasta.write_text(f">query\n{query_seq}\n", encoding="utf-8")

            emb_path = tmp / "db_embeddings.npy"
            ids_path = tmp / "db_ids.txt"
            build_db_embeddings(
                fasta_path=db_fasta,
                output_embeddings_path=emb_path,
                output_ids_path=ids_path,
                output_metadata_jsonl=None,
                embedder_name="aa_onehot",
                device="cpu",
                dtype="fp32",
                log_every=100,
            )

            index_path = tmp / "db.index"
            build_faiss_index(
                embeddings_path=emb_path,
                output_index_path=index_path,
                index_type="flat_ip",
                normalize=True,
                add_batch_size=1024,
            )

            offsets_path = tmp / "db_ids.offsets.u64"
            build_ids_offsets(ids_path, offsets_path)

            workdir = tmp / "workdir"
            args = argparse.Namespace(
                query_fasta=str(query_fasta),
                target_id="target1",
                workdir=str(workdir),
                index_path=str(index_path),
                ids_path=str(ids_path),
                ids_offsets_path=str(offsets_path),
                sequence_fasta=str(db_fasta),
                sequence_sqlite=None,
                build_sequence_sqlite_if_missing=False,
                query_embedding_path=None,
                retrieval_embedder="aa_onehot",
                retrieval_device="cpu",
                alignment_embedder="aa_onehot",
                alignment_device="cpu",
                normalize_query=True,
                top_k=5,
                top_k_prime=5,
                length_ratio_low=0.5,
                length_ratio_high=1.5,
                scale=5.0,
                bias=-1.0,
                gap_open=-2.0,
                gap_extend=-0.5,
                min_query_coverage=0.2,
                min_aligned_query_len=3,
                min_score_density=0.1,
                max_gap_frac=0.95,
                max_rows=4,
                save_alignment_jsonl=None,
                run_openfold=False,
                dry_run_openfold=False,
                openfold_python="python",
                run_pretrained_path="run_pretrained_openfold.py",
                template_mmcif_dir=None,
                config_preset="model_3_ptm",
                model_device="cpu",
                openfold_checkpoint_path=None,
                jax_param_path=None,
                skip_relaxation=True,
                openfold_extra_args=None,
            )

            report = run_pipeline(args)

            align_dir = Path(report["paths"]["alignment_dir"])
            self.assertTrue((align_dir / "bfd_uniclust_hits.a3m").exists())
            self.assertTrue((align_dir / "uniref90_hits.sto").exists())
            self.assertTrue((align_dir / "mgnify_hits.sto").exists())
            self.assertTrue((align_dir / "hhsearch_output.hhr").exists())
            self.assertGreaterEqual(report["gated_rows"], 1)

            query_fasta_out = Path(report["paths"]["query_fasta"])
            from openfold.data.data_pipeline import DataPipeline

            dp = DataPipeline(template_featurizer=None)
            features = dp.process_fasta(str(query_fasta_out), str(align_dir))

            self.assertIn("msa", features)
            self.assertIn("deletion_matrix_int", features)
            self.assertEqual(features["msa"].shape[1], len(query_seq))
            self.assertEqual(features["deletion_matrix_int"].shape[1], len(query_seq))


if __name__ == "__main__":
    unittest.main()
