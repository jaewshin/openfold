Analysis of the SCOPe 40 dataset
------

SCOPe release 2.01 was used in this study, in consistency with the Foldseek paper.

SCOPe 40 data were downloaded from the Foldseek repo:

- https://wwwuser.gwdguser.de/~compbiol/foldseek/

Classification information was retrieved from:

- https://scop.berkeley.edu/downloads/parse/dir.des.scope.2.01-stable.txt

Execute `get_truth.py` to obtain a list of ground-truth matches per domain pair per classification unit (1 - same unit; 2 - otherwise). This will generate `truth.tsv`.

Place the raw results in the `results` folder.

Execute `merge_tables.py` to combine the results into one table file `results.tsv`.

Execute `plot.ipynb` to analyze the results and generate plots.
