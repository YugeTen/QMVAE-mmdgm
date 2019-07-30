
import torch
from torch.utils.data import Dataset
from torchvision import transforms, models, datasets
import torch.nn as nn


import io
import json
import os
import pickle
import numpy as np

from collections import defaultdict
from gensim.models import KeyedVectors, FastText
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter, OrderedDict

class CUBImageFt(Dataset):
    def __init__(self, root_data_dir, split, device):
        """split: 'trainval' or 'test' """

        super().__init__()
        self.data_dir = root_data_dir
        self.data_file = os.path.join(self.data_dir, split)
        self.gen_dir = os.path.join(self.data_dir, 'resnet101_2048')
        self.gen_ft_file = os.path.join(self.gen_dir, '{}.ft'.format(split))
        self.gen_data_file = os.path.join(self.gen_dir, '{}.data'.format(split))
        self.split = split

        tx = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ])
        self.dataset = datasets.ImageFolder(self.data_file, transform=tx)

        os.makedirs(self.gen_dir, exist_ok=True)
        if not os.path.exists(self.gen_ft_file):
            print("Data file not found for CUB image features at {}. Creating new... (this may take a while)".
                  format(self.gen_ft_file))
            self._create_ft_mat(device)

        else:
            self._load_ft_mat()

    def __len__(self):
        return len(self.ft_mat)

    def __getitem__(self, idx):
        return self.ft_mat[idx]

    def _load_ft_mat(self):
        self.ft_mat = torch.load(self.gen_ft_file)

    def _load_data(self):
        self.data_mat = torch.load(self.gen_data_file)

    def _create_ft_mat(self, device):
        resnet = models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.model = nn.Sequential(*modules)
        self.model.eval()

        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}

        loader = torch.utils.data.DataLoader(self.dataset, batch_size=256,
                                             shuffle=False, **kwargs)
        with torch.no_grad():
            ft_mat = torch.cat([self.model(data[0]).squeeze() for data in loader])

        torch.save(ft_mat, self.gen_ft_file)
        del ft_mat

        data_mat = torch.cat([data[0].squeeze() for data in loader])
        torch.save(data_mat, self.gen_data_file)

        self._load_ft_mat()



class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first encountered."""

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)

class CUBSentences(Dataset):

    def __init__(self, root_data_dir, split, transform=None, **kwargs):
        """split: 'trainval' or 'test' """

        super().__init__()
        self.data_dir = root_data_dir
        self.split = split
        self.max_sequence_length = kwargs.get('max_sequence_length', 64)
        self.min_occ = kwargs.get('min_occ', 3)
        self.pre_emb = kwargs.get("use_pretrained_emb", False)
        self.transform = transform

        if self.pre_emb:
            self.model_path = os.path.join(root_data_dir, "lang_emb/wiki-news-300d-1M.vec")
            _url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"
            assert os.path.exists(self.model_path), "Pretrained embedding vector doesn't exist! " \
                                                    "Please download file from {} and extract under" \
                                                    "{}/lang_emb".format(_url, root_data_dir)
            self.gen_dir = os.path.join(self.data_dir, 'wiki_oc:{}_sl:{}'.format(self.min_occ,
                                                                                 self.max_sequence_length))
        else:
            self.ft_size, self.ft_w = kwargs.get("ft_size", 300), kwargs.get("ft_window", 4)
            self.model_path = os.path.join(root_data_dir, "lang_emb/fasttext_s:{}_w:{}_mc:{}.model".
                                           format(self.ft_size, self.ft_w, self.min_occ))
            self.gen_dir = os.path.join(self.data_dir, "oc:{}_sl:{}_s:{}_w:{}".
                                        format(self.min_occ, self.max_sequence_length,
                                               self.ft_size, self.ft_w))

        if split == 'train':
            self.raw_data_path = os.path.join(self.data_dir, 'text_trainvalclasses.txt')
        elif split == 'test':
            self.raw_data_path = os.path.join(self.data_dir, 'text_testclasses.txt')
        else:
            raise Exception("Only train or test split is available")

        os.makedirs(self.gen_dir, exist_ok=True)
        self.data_file = 'cub.{}.s{}'.format(split, self.max_sequence_length)
        self.vocab_file = 'cub.vocab'
        self.emb_file = 'cub.emb'

        if not os.path.exists(os.path.join(self.gen_dir, self.data_file)):
            print("Data file not found for {} split at {}. Creating new... (this may take a while)".
                  format(split.upper(), os.path.join(self.gen_dir, self.data_file)))
            self._create_data()

        else:
            self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[str(idx)]['idx']
        if self.transform is not None:
            sent = self.transform(sent)
        return sent, self.data[str(idx)]['length']

    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def pad_idx(self):
        return self.w2i['<pad>']

    @property
    def eos_idx(self):
        return self.w2i['<eos>']

    @property
    def unk_idx(self):
        return self.w2i['<unk>']

    def get_w2i(self):
        return self.w2i

    def get_i2w(self):
        return self.i2w

    def _load_data(self, vocab=True):
        with open(os.path.join(self.gen_dir, self.data_file), 'rb') as file:
            self.data = json.load(file)

        if vocab:
            self._load_vocab()

    def _load_vocab(self):
        if not os.path.exists(os.path.join(self.gen_dir, self.vocab_file)):
            self._create_vocab()
        with open(os.path.join(self.gen_dir, self.vocab_file), 'r') as vocab_file:
            vocab = json.load(vocab_file)
        self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _create_data(self):

        if self.split == 'train' and not os.path.exists(os.path.join(self.gen_dir, self.vocab_file)):
            self._create_vocab()
        else:
            self._load_vocab()

        with open(self.raw_data_path, 'r') as file:
            text = file.read()
            sentences = sent_tokenize(text)

        data = defaultdict(dict)
        pad_count = 0

        for i, line in enumerate(sentences):
            words = word_tokenize(line)

            tok = words[:self.max_sequence_length - 1]
            tok = tok + ['<eos>']
            length = len(tok)
            if self.max_sequence_length > length:
                tok.extend(['<pad>'] * (self.max_sequence_length - length))
                pad_count += 1
            idx = [self.w2i.get(w, self.w2i['<exc>']) for w in tok]

            id = len(data)
            data[id]['tok'] = tok
            data[id]['idx'] = idx
            data[id]['length'] = length

        print("{} out of {} sentences are truncated with max sentence length {}.".
              format(len(sentences) - pad_count, len(sentences), self.max_sequence_length))
        with io.open(os.path.join(self.gen_dir, self.data_file), 'wb') as data_file:
            data = json.dumps(data, ensure_ascii=False)
            data_file.write(data.encode('utf8', 'replace'))

        self._load_data(vocab=False)

    def _create_vocab(self):

        assert self.split == 'train', "Vocablurary can only be created for training file."

        with open(self.raw_data_path, 'r') as file:
            text = file.read()
            sentences = sent_tokenize(text)

        occ_register = OrderedCounter()
        w2i = dict()
        i2w = dict()

        special_tokens = ['<exc>', '<pad>', '<eos>']
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        texts = []
        unq_words = []
        unk_words = []

        for i, line in enumerate(sentences):
            words = word_tokenize(line)
            occ_register.update(words)
            texts.append(words)

        if self.pre_emb:
            model = KeyedVectors.load_word2vec_format(self.model_path)
        else:
            if os.path.exists(self.model_path):
                model = FastText.load(self.model_path)
            else:
                model = FastText(size=self.ft_size, window=self.ft_w, min_count=self.min_occ)
                model.build_vocab(sentences=texts)
                model.train(sentences=texts, total_examples=len(texts), epochs=0)
                model.save(self.model_path)

        base = np.ones((300,), dtype=np.float32)
        emb = [base * (i - 1) for i in range(len(special_tokens))]

        for w, occ in occ_register.items():
            if occ > self.min_occ and w not in special_tokens:
                i2w[len(w2i)] = w
                w2i[w] = len(w2i)
                if self.pre_emb:
                    if w in model.vocab:
                        emb.append(model[w])
                    else:
                        emb.append(emb[0])
                        unk_words.append(w)
                else:
                    emb.append(model[w])
            else:
                unq_words.append(w)

        assert len(w2i) == len(i2w) == len(emb)
        emb = np.array(emb)
        # print(emb.min())
        # print(emb.max())
        emb = (emb - emb.min()) / (emb.max() - emb.min())

        print("Vocablurary of {} keys created, {} words are excluded, {} "
              "words not in embedding dictionary."
              .format(len(w2i), len(unq_words), len(unk_words)))

        vocab = dict(w2i=w2i, i2w=i2w)
        with io.open(os.path.join(self.gen_dir, self.vocab_file), 'wb') as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode('utf8', 'replace'))

        with open(os.path.join(self.gen_dir, self.emb_file), 'wb') as emb_file:
            pickle.dump(np.array(emb), emb_file)

        with open(os.path.join(self.gen_dir, 'cub.unique'), 'wb') as unq_file:
            pickle.dump(np.array(unq_words), unq_file)

        with open(os.path.join(self.gen_dir, 'cub.unknown'), 'wb') as unknown_file:
            pickle.dump(np.array(unk_words), unknown_file)

        with open(os.path.join(self.gen_dir, 'cub.all'), 'wb') as a_file:
            pickle.dump(occ_register, a_file)

        self._load_vocab()