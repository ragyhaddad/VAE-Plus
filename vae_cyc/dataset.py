import torch 
import numpy as np
from .vocab import Vocab
from torch.nn.utils.rnn import pad_sequence 
from torch.utils.data import IterableDataset
# from SmilesPE.pretokenizer import atomwise_tokenizer 

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
    
    def __getitem__(self, idx):
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

class TransformerSMILESDataset:
    def __init__(self, sequences, vocab, max_len=0, pretokenize=False, dynamic_padding=False):
        self.sequences = np.array(sequences).astype(np.string_) 
        self.vocab = vocab 
        self.max_len = max_len 
        self.pad_idx = vocab.pad_idx
        self.pretokenize = pretokenize
        self.dynamic_padding = dynamic_padding
        if pretokenize:
            self.tokenize(self.sequences)
    
    def tokenize(self, sequences):
        from tqdm import tqdm
        self.tokens = []
        if self.pretokenize:
            for seq in tqdm(sequences):
                seq = str(seq, encoding='utf-8')
                seq = self.vocab.encode_special(seq)
                tokens = [self.vocab.char2idx[i] for i in seq]
                self.tokens.append(tokens)
    
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.pretokenize:
            tokens = self.tokens[idx]
        else:
            seq = self.sequences[idx]
            seq = str(seq, encoding='utf-8')
            seq = self.vocab.encode_special(seq)
            tokens = [self.vocab.char2idx[i] for i in seq]
        with_bos = [self.vocab.char2idx[self.vocab.sos_token]] + tokens 
        with_eos = tokens + [self.vocab.char2idx[self.vocab.eos_token]]

        if self.dynamic_padding is False:
            with_bos += [self.pad_idx] * (self.max_len - len(with_bos))
            with_eos += [self.pad_idx] * (self.max_len - len(with_eos))
        
            return torch.tensor(with_bos), torch.tensor(with_eos)
        else:
            return torch.tensor(with_bos), torch.tensor(with_eos)
    
    def collate(self, batch):
        with_bos, with_eos = zip(*batch)
        if self.dynamic_padding is False:
            with_bos = torch.stack(with_bos) 
            with_eos = torch.stack(with_eos)
            attention_masks = (with_bos == self.pad_idx).bool() # this is the padding mask, True for padding False for non-padding
            return with_bos, with_eos, attention_masks
        else:
            with_bos = pad_sequence(with_bos, batch_first=True, padding_value=self.pad_idx)
            with_eos = pad_sequence(with_eos, batch_first=True, padding_value=self.pad_idx)
            attention_masks = (with_bos == self.pad_idx).bool()
            return with_bos, with_eos, attention_masks

class TransformerMemapDataset:
    def __init__(self, file_name, vocab):
        self.file_name = file_name
        self.vocab = vocab  
        self.metadata = None
        self.sos_idx = vocab.sos_idx
        self.eos_idx = vocab.eos_idx
        self.read_metadata(self.file_name + '.json')
        self.read_memmap(self.file_name + '.dat')
    
    def read_metadata(self, file_path):
        import json
        with open(file_path, 'r') as f:
            self.metadata = json.load(f)
            print(self.metadata)
        self.dtype = self.metadata['dtype']
        self.shape = tuple(self.metadata['shape'])
        self.max_len = self.shape[1] + 1
    
    def read_memmap(self, file_path):
        self.memmap = np.memmap(file_path, dtype=self.dtype, mode='r', shape=self.shape)
    
    def __len__(self):
        return self.shape[0]
    
    def __getitem__(self, idx):
        sample = self.memmap[idx]
        with_bos = sample[sample != self.vocab.eos_idx]
        with_eos = sample[sample != self.vocab.sos_idx]
        # padding 
        # with_bos = np.pad(with_bos, (0, self.max_len - len(with_bos)), mode='constant', constant_values=self.vocab.pad_idx)
        # with_eos = np.pad(with_eos, (0, self.max_len - len(with_eos)), mode='constant', constant_values=self.vocab.pad_idx)
        return torch.tensor(with_bos).long(), torch.tensor(with_eos).long()

    def collate(self, batch):
        with_bos, with_eos = zip(*batch)
        with_bos = torch.stack(with_bos)
        with_eos = torch.stack(with_eos)
        with_bos = pad_sequence(with_bos, batch_first=True, padding_value=self.vocab.pad_idx)
        with_eos = pad_sequence(with_eos, batch_first=True, padding_value=self.vocab.pad_idx)



        masks = (with_bos == self.vocab.pad_idx).bool()
        
        # print(with_bos.shape, with_eos.shape, masks.shape)
        return with_bos, with_eos, masks

