import torch 
import numpy as np
from .vocab import Vocab

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, vocab, target_col, max_len, verbose=True):
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
        self.all_seqs = np.array(df[target_col].values).astype(np.string_) 
        # self.tokenize(df)
    
    def __getitem__(self, idx):
        # seq = self.tokens[idx]
        seq = self.all_seqs[idx]
        seq = str(seq, encoding='utf-8')
        seq = [self.char2idx[c] for c in seq]
        with_bos = torch.tensor([self.char2idx[self.vocab.sos_token]] + seq).long()
        with_eos = torch.tensor(seq + [self.char2idx[self.vocab.eos_token]]).long()
        return with_bos, with_eos
    
    def __len__(self):
        return len(self.all_seqs)
    
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

class TransformerDataset:
    def __init__(self, sequences, vocab, max_len=100):
        self.sequences = sequences
        self.vocab = vocab 
        self.max_len = max_len 
    
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq = self.vocab.handle_special(seq)
        tokens = [self.vocab.char2idx[i] for i in seq]
        
        with_bos = [self.vocab.char2idx[self.vocab.sos_token]] + tokens
        with_eos = tokens + [self.vocab.char2idx[self.vocab.eos_token]]
        with_bos += [self.vocab.char2idx[self.vocab.pad_token]] * (self.max_len - len(with_bos))
        with_eos += [self.vocab.char2idx[self.vocab.pad_token]] * (self.max_len - len(with_eos))
        attention_mask = [1 if t != self.vocab.char2idx[self.vocab.pad_token] else 0 for t in with_bos]
        return torch.tensor(with_bos), torch.tensor(with_eos), torch.tensor(attention_mask).float()

    def collate(self, batch):
        with_bos, with_eos, masks = zip(*batch)
        with_bos = torch.stack(with_bos) 
        with_eos = torch.stack(with_eos)
        masks = torch.stack(masks)
        return with_bos, with_eos, masks