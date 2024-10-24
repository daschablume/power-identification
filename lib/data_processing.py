from typing import Optional, Callable
from typing_extensions import Self
from collections.abc import Iterable
from collections import defaultdict
from pathlib import Path

import csv
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tqdm import tqdm

import numpy as np
from scipy.sparse import csr_matrix, issparse

import gensim
import gensim.downloader as api
from nltk.tokenize import word_tokenize
from transformers import DistilBertTokenizer, DistilBertModel  

class PositionalEncoder(BaseEstimator, TransformerMixin):
    """Positional encoder to encode text into positional vectors. Used for RNN models.
    """
    def __init__(
            self, 
            vocabulary: Optional[Iterable] = None,
            tokenizer: Optional[Callable] = None,
            no_progress_bar: bool = True,
        ) -> None:
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer or self.build_tokenizer()
        self.max_sentence_length_ = 0
        self.no_progress_bar = no_progress_bar

    def token_to_index(self, token):
        if token in self.vocabulary:
            return self.vocabulary[token]
        else:
            return self.vocabulary['<UNK>']

    def index_to_token(self, index):
        return list(self.vocabulary.keys())[index]

    def fit(self, X: list[str], y: Optional[list] = None):
        """Learn vocabulary from provided sentences and labels
        """
        vocabulary: set = set()
        max_sentence_length = 0

        # Iterate through sentences in the input, tokenize, and update vocabulary
        for sentence in tqdm(X, total=len(X), desc="Fit\t\t", unit="sample", disable=self.no_progress_bar):
            tokens = self.tokenizer(sentence)
            if len(tokens) > max_sentence_length:
                max_sentence_length = len(tokens)

            if self.vocabulary is None:
                for token in tokens:
                    vocabulary.add(token)
        
        if self.vocabulary is None:
            vocab_list = ['<UNK>'] + list(vocabulary) + ['<PAD>']
            self.vocabulary = {word: idx for idx, word in enumerate(vocab_list)}

        self.max_sentence_length_ = max_sentence_length

        return self

    def transform(self, X, y: Optional[list] = None):

        crow, col, token_val = [], [], []

        # Iterate through sentences, construct the necessary arrays to create CSR sparse matrix
        for i, text in tqdm(enumerate(X), total=len(X), desc="Transform\t", unit="sample", disable=self.no_progress_bar):
            tokens = self.tokenizer(text)
            crow.append(i * self.max_sentence_length_)

            for j, token in enumerate(tokens):
                col.append(j)

                # Append index of tokens and tags to the val arrays
                if token in self.vocabulary:
                    token_val.append(self.token_to_index(token))
                else:
                    token_val.append(self.token_to_index("<UNK>"))

            # Add padding to make all sentences have the same length
            padding_amt = self.max_sentence_length_ - len(tokens)
            token_val += [self.token_to_index("<PAD>")] * padding_amt

            # Column index of the paddings runs from (len of tokens) to max_sentence_length
            col += list(range(len(tokens), self.max_sentence_length_))

        assert len(token_val) == self.max_sentence_length_ * len(X), \
            f"Length of token_val is incorrect: {len(token_val)} != {self.max_sentence_length_ * len(X)}"

        # Construct sparse matrices
        mat_size = (len(X), self.max_sentence_length_)
        tokens_sparse = torch.sparse_csr_tensor(crow, col, token_val, size=mat_size, dtype=torch.long)

        return tokens_sparse.to_dense()
    
    def build_tokenizer(self):
        def simple_tokenizer(sentence):
            words: list = sentence.split(' ')
            new_words = []

            for word in words:
                # If there is no punctuation in the word, add to the final list
                # Else, split the words further at the punctuations
                if all([char.isalnum() for char in word]):
                    new_words.append(word)

                else:
                    tmp = ''
                    # Iterate through characters. When encounter a punctuation,
                    # add the previous characters as a word, then add the punctuation
                    for char_idx, char in enumerate(word):
                        if char.isalnum():
                            tmp += char
                            if char_idx == len(word) - 1:
                                new_words.append(tmp)
                        else:
                            if char_idx > 0:
                                new_words.append(tmp)
                            new_words.append(char)
                            tmp = ''
            return new_words
        return simple_tokenizer
    
    def fit_transform(self, X, y: Optional[list] = None):
        self.fit(X, y)
        tokens = self.transform(X, y)
        return tokens


class RawDataset():
    """Class to hold raw data load directly from the tsv files.
    """
    def __init__(self, ids: list[str], speakers: list[str], texts: list[str], labels: list[int]) -> None:
        assert len(ids) == len(speakers) == len(texts) == len(labels), "All arrays must have the same length"
        self.ids = ids
        self.speakers = speakers
        self.texts = texts
        self.labels = labels

    def subset(self, index_list: list[int]):

        data = RawDataset(
            [self.ids[idx] for idx in index_list],
            [self.speakers[idx] for idx in index_list],
            [self.texts[idx] for idx in index_list],
            [self.labels[idx] for idx in index_list],
        )
        
        return data

    def __getitem__(self, index: int):
        return (self.ids[index], self.speakers[index], self.texts[index], self.labels[index])

    def __add__(self, other: Self):
        return RawDataset(
            self.ids + other.ids,
            self.speakers + other.speakers,
            self.texts + other.texts,
            self.labels + other.labels
        )

    def __iter__(self):
        for data in zip(self.ids, self.speakers, self.texts, self.labels):
            yield data

    def __len__(self):
        return len(self.ids)


class EncodedDataset(Dataset):
    """Custom Dataset object to hold parliament debate data. Each item in the dataset
    is a tuple of (input tensor, label)
    """
    def __init__(
            self, 
            inputs: torch.Tensor, 
            labels: torch.Tensor, 
        ) -> None:
        super().__init__()
        assert len(inputs) == len(labels), "Inputs and labels have different length"
        self.data_ = list(zip(inputs, labels))

    def __len__(self):
        return len(self.data_)

    def __getitem__(self, index):
        return self.data_[index]
    
    def __iter__(self):
        for data in self.data_:
            yield data


def load_data(file_path_list: list[Path | str], text_head: str = 'text') -> RawDataset:
    """Load the Parliament Debate dataset. 

    Parameters
    ----------
    file_path_list : list
        List of path to the files you want to load
    text_head : str, optional
        Name of the text column, either 'text' or 'text_en', by default 'text'

    Returns
    -------
    RawDataset
        Returns a RawDataset object containing: text ID, speaker ID, text, label
    """

    for file_path in file_path_list:
        if isinstance(file_path, str):
            file_path = Path(file_path)
            
        data = RawDataset([], [], [], [])
        tmp_data = _load_one(file_path=file_path, text_head=text_head)
        data += tmp_data

    return data


def _load_one(file_path, encoding: str = 'utf-8', text_head: str = 'text') -> RawDataset:
    """Load one file and return """
    
    ids         : list[str] = []
    speakers    : list[str] = []
    texts       : list[str] = []
    labels      : list[int] = []

    with open(file_path, "rt", encoding=encoding) as f:
        csv_r = csv.DictReader(f, delimiter="\t")
        
        for row in csv_r:
            ids.append(row.get('id'))
            speakers.append(row.get('speaker'))
            texts.append(row.get(text_head))
            labels.append(int(row.get('label', -1)))
        
    return RawDataset(ids, speakers, texts, labels)


def split_data(data: RawDataset, test_size=0.2, random_state=None) -> tuple[RawDataset, RawDataset]:
    """Return train-test sets divided by speakers, so that speakers of the train and test set do not overlap"""
    speaker_indices = defaultdict(list)

    # Inverted index: {speaker: [idx1, idx2]}
    for idx, speaker in enumerate(data.speakers):
        speaker_indices[speaker].append(idx)

    # Split list of (indices list)
    # speaker_indices.values() = list of list of indices
    train_indices_lst, test_indices_lst = train_test_split(
        list(speaker_indices.values()), 
        test_size=test_size, 
        random_state=random_state
    )

    # Flatten the list of list of indices
    train_indices = [idx for lst in train_indices_lst for idx in lst]
    test_indices = [idx for lst in test_indices_lst for idx in lst]
    
    return data.subset(train_indices), data.subset(test_indices)


def encode_torch_data(raw_data: RawDataset, encoder: PositionalEncoder | TfidfVectorizer):
    """Convenience function to create the encoded dataset compatible with torch models"""
    # Encode text
    enc_texts_csr = encoder.transform(raw_data.texts)  
    
    if isinstance(enc_texts_csr, csr_matrix):
        inputs = torch.from_numpy(enc_texts_csr.todense()).float()
    else:
        inputs = enc_texts_csr.to_dense()
    
    # Convert labels to tensor
    labels = torch.tensor(raw_data.labels)

    return EncodedDataset(inputs, labels)

def create_dataloader(X_embedded, y, batch_size=32, shuffle=False):
    # Check if the input is a sparse matrix and convert it to dense if needed
    if issparse(X_embedded):
        X_embedded = X_embedded.toarray()

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_embedded, dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y), dtype=torch.long)
    
    # Create dataset and dataloader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader

def get_embeddings(method, X_train, X_test, max_features=None):
    if method == 'Binary_BoW' or method == 'TF-IDF':
        vectorizer = CountVectorizer(binary=True, max_features=max_features) if method == 'Binary_BoW' else TfidfVectorizer(max_features=max_features)
        X_train_embedded = vectorizer.fit_transform(X_train)
        X_test_embedded = vectorizer.transform(X_test)
        return X_train_embedded, X_test_embedded

    elif method == 'Word2Vec':
        # Ensure X_train, X_test are lists of tokenized sentences
        X_train_tokenized = [word_tokenize(sentence.lower()) for sentence in X_train]
        X_test_tokenized = [word_tokenize(sentence.lower()) for sentence in X_test]
        
        # Load Word2Vec embeddings using Gensim
        model = gensim.models.Word2Vec(sentences=X_train_tokenized, vector_size=max_features, window=5, min_count=1)
        X_train_embedded = np.array([
            np.mean([model.wv[word] for word in sentence if word in model.wv] or [np.zeros(max_features)], axis=0) 
            for sentence in X_train_tokenized
        ])
        X_test_embedded = np.array([
            np.mean([model.wv[word] for word in sentence if word in model.wv] or [np.zeros(max_features)], axis=0) 
            for sentence in X_test_tokenized
        ])
        return X_train_embedded, X_test_embedded

    elif method == 'GloVe':
        # Ensure X_train, X_text are lists of tokenized sentences
        X_train_tokenized = [word_tokenize(sentence.lower()) for sentence in X_train]
        X_test_tokenized = [word_tokenize(sentence.lower()) for sentence in X_test]

        # Load GloVe embeddings using Gensim
        glove_embeddings = load_glove_embeddings_from_gensim(max_features)
        vocab = {word: i for i, word in enumerate(set(word for sentence in X_train_tokenized for word in sentence))}
        embedding_dim = len(next(iter(glove_embeddings.values())))  # Dimension of the GloVe embeddings
        embedding_matrix = get_glove_embedding_matrix(vocab, glove_embeddings, embedding_dim)
        
        X_train_embedded = np.array([
            np.mean([embedding_matrix[vocab[word]] for word in sentence if word in vocab] or [np.zeros(embedding_dim)], axis=0) 
            for sentence in X_train_tokenized
        ])
        X_test_embedded = np.array([
            np.mean([embedding_matrix[vocab[word]] for word in sentence if word in vocab] or [np.zeros(embedding_dim)], axis=0) 
            for sentence in X_test_tokenized
        ])
        return X_train_embedded, X_test_embedded

    elif method == 'DistilBERT':
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        X_train_embedded = generate_transformer_embeddings(tokenizer, model, X_train, max_length=max_features, device=device)
        X_test_embedded = generate_transformer_embeddings(tokenizer, model, X_test, max_length=max_features, device=device)
        return X_train_embedded, X_test_embedded

    else:
        raise ValueError(f"Unsupported embedding method: {method}")

def load_glove_embeddings_from_gensim(max_features=None):
    # Load pre-trained GloVe embeddings directly from Gensim
    glove_model = api.load("glove-wiki-gigaword-300")  # This loads GloVe with 300d vectors
    
    # Limit to max_features
    glove_embeddings = {}
    for i, word in enumerate(glove_model.key_to_index):
        if max_features and i >= max_features:
            break
        glove_embeddings[word] = glove_model[word]
    
    return glove_embeddings

def get_glove_embedding_matrix(vocab, glove_embeddings, embedding_dim):
    embedding_matrix = np.zeros((len(vocab), embedding_dim))
    for word, idx in vocab.items():
        embedding_vector = glove_embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector
    return embedding_matrix

def generate_transformer_embeddings(tokenizer, model, texts, max_length=128, device="cpu"):
    model = model.to(device)  # Move the model to the specified device (GPU or CPU)
    model.eval()  # Set model to evaluation mode
    embeddings = []
    
    for text in texts:
        # Tokenize the text
        encoded_input = tokenizer(text, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt").to(device)  # Move input to device
        
        # Forward pass to get hidden states
        with torch.no_grad():
            output = model(**encoded_input)
        
        # Use the embeddings from the last hidden state
        hidden_states = output.last_hidden_state
        sentence_embedding = torch.mean(hidden_states, dim=1).squeeze().cpu().numpy()  # Move result back to CPU if necessary
        embeddings.append(sentence_embedding)
    
    return np.array(embeddings)

