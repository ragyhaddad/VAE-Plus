import re 
from SmilesPE.pretokenizer import atomwise_tokenizer


        

class Vocab:
    def __init__(self, df, smiles_col, char2idx={}, idx2char={}):
        """
        Vocab module for all VAE models

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing SMILES strings
        smiles_col : str
            Column name of SMILES strings
        char2idx : dict
            Dictionary mapping characters to indices
        idx2char : dict
            Dictionary mapping indices to characters
        """
        self.sos_token = '<sos>'
        self.eos_token = '<eos>'
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.smiles_col = smiles_col
        self.special_tokens = [self.sos_token, self.eos_token, self.pad_token, self.unk_token]
        self.special_atom = {'Cl': 'Q', 'Br': 'W', '[nH]': 'X', '[H]': 'Y'}
        self.step = len(self.char2idx)

        # if df is not None:
        #     self.extract_charset(df)

    @property
    def sos_idx(self):
        """
        sos token index
        """
        return self.char2idx[self.sos_token]

    @property
    def eos_idx(self):
        """
        eos token index
        """
        return self.char2idx[self.eos_token]

    @property
    def unk_idx(self):
        """
        unk token index
        """
        return self.char2idx[self.unk_token]

    @property
    def pad_idx(self):
        """
        pad token index
        """
        return self.char2idx[self.pad_token]

    @property
    def vocab_size(self):
        """
        Returns
        -------
        int
            Number of unique characters in the charset
        """
        return len(self.char2idx)

    def extract_charset(self, df):
        """
        Extract charset from SMILES strings

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing SMILES strings

        """
        from tqdm import tqdm

        print('extracting charset..')
        
        for c in self.special_tokens:
            if c not in self.char2idx:
                self.char2idx[c] = self.step
                self.step += 1
        all_smi = df[self.smiles_col].values.flatten()
        lengths = []
        for _, smi in enumerate(tqdm(all_smi)):
            smi = smi.replace('Cl', 'Q')
            smi = smi.replace('Br', 'W')
            smi = smi.replace('[nH]', 'X')
            smi = smi.replace('[H]', 'Y')
            
            lengths.append(len(smi))

            for c in smi:
                if c not in self.char2idx:
                    self.char2idx[c] = self.step
                    self.step += 1
        self.idx2char = {v:k for k,v in self.char2idx.items()}
    
    def save(self, file_path):
        import json
        with open(file_path,'w') as f:
            json.dump(self.char2idx, f)
    
    @classmethod
    def load(cls, file_path):
        import json
        with open(file_path,'r') as f:
            char2idx = json.load(f)
        idx2char = {v:k for k,v in char2idx.items()}
        return cls(None, None, char2idx, idx2char)

    def encode_special(self, smi):
        smi = smi.replace('Cl', 'Q')
        smi = smi.replace('Br', 'W')
        smi = smi.replace('[nH]', 'X')
        smi = smi.replace('[H]', 'Y')
        return smi 

    def decode_special(self, smi):
        smi = smi.replace('Q', 'Cl')
        smi = smi.replace('W','Br')
        smi = smi.replace('X','[nH]')
        smi = smi.replace('Y','[H]')
        return smi
    
    def tokenize(self, smi):
        smi_split = self.smi_tokenizer(smi)
        return smi_split
        

class AtomVocab(Vocab):
    def __init__(self, df, smiles_col, char2idx={}, idx2char={}):
        super().__init__(df, smiles_col, char2idx, idx2char)
        
    def extract_charset(self, df):
        print('extracting charset with atomwise tokenizer..')
        from SmilesPE.pretokenizer import atomwise_tokenizer

        """
        Extract charset from SMILES strings

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing SMILES strings

        """
        from tqdm import tqdm
        self.max_len = 0
        i = 0
        for c in self.special_tokens:
            if c not in self.char2idx:
                self.char2idx[c] = i
                self.idx2char[i] = c
                i += 1
        all_smi = df[self.smiles_col].values.flatten()
        for j, smi in enumerate(tqdm(all_smi)):
            smi = atomwise_tokenizer(smi)
            if len(smi) > self.max_len:
                self.max_len = len(smi)

            for c in smi:
                if c not in self.char2idx:
                    if i == 0:
                        print(c)
                    self.char2idx[c] = i
                    self.idx2char[i] = c
                    i += 1

class AminoAcidVocab(Vocab):
    def __init__(self, df=None, target_col=None):
        super().__init__(df, target_col)
        self.extract_charset(df=df)
    
    ## Set up the vocab with all 20 known AAs and special tokens
    def extract_charset(self, df):
        i = 0
        for c in self.special_tokens:
            self.char2idx[c] = i
            self.idx2char[i] = c
            i += 1
        for aa in 'ACDEFGHIKLMNPQRSTVWYXUZBO':
            self.char2idx[aa] = i
            self.idx2char[i] = aa
            i += 1


class RegexVocab(Vocab):
    def __init__(self, df, smiles_col, char2idx={}, idx2char={}):
        super().__init__(df, smiles_col, char2idx, idx2char)
        self.regex_pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
        # self.regex_pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#||\+|\\\\\/|:||@|\?|>|\*|\$|\%[0–9]{2}|[0–9])"

        self.step = len(char2idx)
        if df is not None:
            self.extract_charset(df)
    
    def extract_charset(self, df):
        print('extracting charset with regex tokenizer..')

        """
        Extract charset from SMILES strings

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing SMILES strings

        """
        from tqdm import tqdm
        self.max_len = 0
        for c in self.special_tokens:
            if c not in self.char2idx:
                self.char2idx[c] = self.step
                self.idx2char[self.step] = c
                self.step += 1
        all_smi = df[self.smiles_col].values.flatten()
        for j, smi in enumerate(tqdm(all_smi)):
            smi = self.smi_tokenizer(smi)
            if len(smi) > self.max_len:
                self.max_len = len(smi)

            for c in smi:
                if c not in self.char2idx:
                    if self.step == 0:
                        print(c)
                    self.char2idx[c] = self.step
                    self.idx2char[self.step] = c
                    self.step += 1


    def smi_tokenizer(self, smi):
        regex = re.compile(self.regex_pattern)
        tokens = [token for token in regex.findall(smi)]
        assert smi == ''.join(tokens)
        return list(tokens)
    
    def tokenize(self, smi):
        smi_split = self.smi_tokenizer(smi)
        return smi_split