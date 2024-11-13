class Vocab:
    def __init__(self, df, target_col, char2idx={}, idx2char={}):
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.target_col = target_col
        self.sos_token = '<sos>'
        self.eos_token = '<eos>'
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.special_tokens = [self.sos_token, self.eos_token, self.pad_token, self.unk_token]
        if df is not None:
            self.build_vocab(df)
    
    @property
    def sos_idx(self):
        return self.char2idx[self.sos_token]
    
    @property
    def eos_idx(self):
        return self.char2idx[self.eos_token]
    
    @property
    def pad_idx(self):
        return self.char2idx[self.pad_token]
    
    @property
    def unk_idx(self):
        return self.char2idx[self.unk_token]
    
    @property
    def vocab_size(self):
        return len(self.char2idx)
    
    def build_vocab(self, df):
        from tqdm import tqdm
        print('extracting charset...')
        i = 0
        for c in self.special_tokens:
            if c not in self.char2idx:
                self.char2idx[c] = i
                self.idx2char[i] = c
                i += 1
        lengths = []
        ## iterate over the rows of the dataframe and use tqdm to show a progress bar
        # show progress in loop
        for j in tqdm(range(len(df))):
            seq = df.iloc[j][self.target_col]
            lengths.append(len(seq))
            for c in seq:
                if c not in self.char2idx:
                    self.char2idx[c] = i
                    self.idx2char[i] = c
                    i += 1

class AminoAcidVocab(Vocab):
    def __init__(self, df=None, target_col=None):
        super().__init__(df, target_col)
        self.build_vocab(df=df)
    
    ## Set up the vocab with all 20 known AAs and special tokens
    def build_vocab(self, df):
        i = 0
        for c in self.special_tokens:
            self.char2idx[c] = i
            self.idx2char[i] = c
            i += 1
        for aa in 'ACDEFGHIKLMNPQRSTVWYXUZBO':
            self.char2idx[aa] = i
            self.idx2char[i] = aa
            i += 1

class TransformerVocab:
    def __init__(self, sequences=None, target_col=None, char2idx={}, idx2char={}):
        self.sos_token = '<sos>' 
        self.eos_token = '<eos>' 
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]
        self.char2idx = char2idx 
        self.idx2char = idx2char
        if sequences != None:
            self.extract_charset(sequences)
        else:
            self.build_vocab_aa()

    def handle_special(self, smi):
        smi = smi.replace('Cl', 'Q')
        smi = smi.replace('Br', 'W')
        smi = smi.replace('[nH]', 'X')
        smi = smi.replace('[H]', 'Y')
        return smi
    def reverse_special(self, smi):
        smi = smi.replace('Q', 'Cl')
        smi = smi.replace('W', 'Br')
        smi = smi.replace( 'X','[nH]')
        smi = smi.replace('Y', '[H]') 
        return smi 
    def extract_charset(self, sequences):
        """
        Extract charset from SMILES strings

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing SMILES strings

        """
        from tqdm import tqdm

        print('extracting charset..')
        i = 0
        for c in self.special_tokens:
            if c not in self.char2idx:
                self.char2idx[c] = i
                self.idx2char[i] = c
                i += 1
        all_smi = sequences
        for _, smi in enumerate(tqdm(all_smi)):
            smi = self.handle_special(smi)
            for c in smi:
                if c not in self.char2idx:
                    self.char2idx[c] = i
                    self.idx2char[i] = c
                    i += 1
    
    def build_vocab_aa(self):
        aa_alphabet = list('ACDEFGHIKLMNPQRSTVWYXUZBO')  # 20 standard amino acids
        all_chars = self.special_tokens + aa_alphabet
        self.char2idx = {token: idx for idx, token in enumerate(all_chars)}
        self.idx2char = {idx: token for token, idx in self.char2idx.items()}