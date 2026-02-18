Create domain list:

cat cath-domain-seqs-S100.fa | grep -m 1000 ^'>' | cut -f3 -d'|' | cut -f1 -d'/' > domain.lst
