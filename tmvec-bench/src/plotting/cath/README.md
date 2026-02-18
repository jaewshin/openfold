Analysis of the CATH S100 dataset
------

CATH release 4.4.0 was used in this study. It is available for download at:

- ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/all-releases/v4_4_0/

Download sequence data:

```bash
wget -O cath-domain-seqs-S100.fa ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/all-releases/v4_4_0/sequence-data/cath-domain-seqs-S100-v4_4_0.fa
```

Download classification data:

```bash
wget -O cath-domain-list-S100.txt ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/all-releases/v4_4_0/cath-classification-data/cath-domain-list-S100-v4_4_0.txt
wget -O cath-names.txt ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/all-releases/v4_4_0/cath-classification-data/cath-names-v4_4_0.txt
```

Extract top 1000 sequences:

```bash
cat cath-domain-seqs-S100.fa | grep -m 1000 ^'>' -A 1 > seqs.fa
```

These sequences will be fed to the models to assess pairwise similarity.

Extract a list of domains:

```bash
cat seqs.fa | cut -f3 -d'|' | cut -f1 -d'/' > domain.lst
```

Execute `get_truth.py` to obtain a list of ground-truth matches per domain pair per classification unit (1 - same unit; 2 - otherwise). This will generate `truth.tsv`.

Place the raw results in the `results` folder.

Execute `merge_tables.py` to combine the results into one table file `results.tsv`.

Execute `plot.ipynb` to analyze the results and generate plots.
