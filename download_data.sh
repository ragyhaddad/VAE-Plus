mkdir -p datasets
wget https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz  -O datasets/uniprot_sprot.fasta.gz  
gunzip datasets/uniprot_sprot.fasta.gz 

wget https://raw.githubusercontent.com/charlesxu90/helm-gpt/refs/heads/main/data/prior/monomer_library.csv -O datasets/monomer_library.csv

wget https://raw.githubusercontent.com/charlesxu90/helm-gpt/refs/heads/main/data/prior/chembl32/biotherapeutics_dict_prot_flt.csv -O datasets/biotherapeutics_dict_prot_flt.csv 

wget https://github.com/charlesxu90/helm-gpt/blob/main/data/prior/CycPeptMPDB/CycPeptMPDB_Peptide_All.csv -O datasets/CycPeptMPDB_Peptide_All.csv

wget https://github.com/charlesxu90/helm-gpt/blob/main/data/prior/prior_data.csv -O datasets/helm_prior.csv