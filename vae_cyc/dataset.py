import torch 
from .vocab import Vocab

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, vocab, target_col, max_len, verbose=True):
        self.df = df
        self.vocab = vocab
        self.target_col = target_col
        self.max_len = max_len
        self.pad_idx = vocab.pad_idx
        self.sos_idx = vocab.sos_idx
        self.eos_idx = vocab.eos_idx
        self.unk_idx = vocab.unk_idx
        self.vocab_size = vocab.vocab_size
        self.char2idx = vocab.char2idx
        self.idx2char = vocab.idx2char
        self.verbose = verbose
        self.tokenize(df)
    
    def __getitem__(self, idx):
        seq = self.tokens[idx]
        with_bos = torch.tensor([self.char2idx[self.vocab.sos_token]] + seq).long()
        with_eos = torch.tensor(seq + [self.char2idx[self.vocab.eos_token]]).long()
        return with_bos, with_eos
    
    def __len__(self):
        return len(self.tokens)
    
    def collate(self, samples):
        with_bos, with_eos = list(zip(*samples))
        lengths = [len(seq) for seq in with_bos]
        with_bos = torch.nn.utils.rnn.pad_sequence(with_bos, batch_first=True, padding_value=self.pad_idx)
        with_eos = torch.nn.utils.rnn.pad_sequence(with_eos, batch_first=True, padding_value=self.pad_idx)
        return with_bos, with_eos, lengths
    
    def get_vocab(self):
        return self.vocab(self.char2idx, self.idx2char)
    
    def tokenize(self, df):
        from tqdm import tqdm
        if self.verbose:
            print('Tokenizing...')
        self.tokens = []
        ## iterate over df targte col and tokenize 
        for i in tqdm(range(len(df))):
            seq = df.iloc[i][self.target_col]
            tokens = [self.char2idx[c] for c in seq]
            self.tokens.append(tokens)
        if self.verbose:
            print('Tokenization complete')
