mkdir -p datasets
wget https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz  -O datasets/uniprot_sprot.fasta.gz  
gunzip datasets/uniprot_sprot.fasta.gz 
