import json
import numpy as np
import os
import pandas as pd
import re
from nltk.stem import PorterStemmer
from pandas.core.frame import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Union

PASSAGE_LEN = 200                       # Chunking parameter for passage length (0, 500)
STEMMER = PorterStemmer()           

class Document:
    """
    A class used to represent a Document (chapter or a passage)

    ...

    Attributes
    ----------
    book : int
        the name of the book the chapter or passage belongs to
    chapter_no : int
        the chapter number of the passage or the chapter itself
    text : List[str]
        the list of all the lines in a chapter or passage
    ps : PorterStemmer
        the stemmer object used to stem the words in the document (global)

    Methods
    -------
    stem()
        Returns the stemmed text
    """

    def __init__(self, book: str, chapter_no: int, text: List[str], ps: PorterStemmer = STEMMER) -> None:
        self.book = book
        self.chapter_no = chapter_no
        self.text = text
        self.ps = ps
        self.stemmed = self.stem()

    def stem(self) -> str:
        text = []
        for line in self.text:
            newline = ''
            for word in line:
                newline += self.ps.stem(word)
            text.append(newline)

        return text

    def __str__(self) -> str:
        return " ".join(self.text)


class Query:
    """
    A class used to represent a Query

    ...

    Attributes
    ----------
    text : str
        the query text 
    ps : PorterStemmer
        the stemmer object used to stem the words in the query (global)

    Methods
    -------
    stem()
        Returns the stemmed text
    """

    def __init__(self, text: str, ps: PorterStemmer = STEMMER) -> None:
        # plain text
        self.text = text
        self.ps = ps

    def stem(self) -> str:
        stemmed = ""
        for word in self.text.split(' '):
            stemmed += self.ps.stem(word) + " "

        return stemmed.strip()

    def __str__(self) -> str:
        return self.text


Corpus = List[Document]


def read_corpus(path: Union[os.PathLike, str], name: str, ps: PorterStemmer = STEMMER) -> Corpus:
    """Return a list of chapters made from the given Bible in the path."""
    
    with open(path) as f:
        bible = json.load(f)

    def preprocess(text):
        p = re.compile("<(/)?\w>")
        text = p.sub("", text)
        text = text.replace("\n", " ")
        return text.lower()

    book = bible[name]
    chapters = list(book.keys())
    corpus = []
    for i, chapter in enumerate(chapters):
        text = book[chapter][1:]
        text = ["\n".join(passage[1:]) for passage in text]
        text = "\n".join(text)
        text = preprocess(text)
        text = text.split(".")
        for line in text:
            line += "."
        corpus.append(Document(chapter, i + 1, text, ps))
    return corpus


class DocumentRetrieval:
    """
    Generate term frequency matrix for a corpus and provide interface for retrieval

    ...

    Attributes
    ----------
    documents: Corpus
        the corpus to index (read from storage if None)
    vectorizer : TfidfVectorizer 
        the algorithm used to build the frequency matrix (Tf-idf)

    Methods
    -------
    retrieve(query)
        Returns a list of candidate documents most similar to the query
    """

    def __init__(self, documents: Corpus = None) -> None:

        if not documents:
            path = os.path.join("kjv.json")
            name = "KingJamesVersion"
            self.documents = read_corpus(path, name)
        else:
            self.documents = documents

        self.vectorizer = TfidfVectorizer(stop_words='english',
                                          binary=True,
                                          ngram_range=(1, 2),
                                          sublinear_tf=True)
        self.inverted_doc = self.corpus_vectorizer()
        self.vocab_len = self.inverted_doc.shape[0]
        self.doc_len = len(self.documents)

    def corpus_vectorizer(self) -> DataFrame:
        """Return a frequency matrix built from the member corpus."""

        docs = [" ".join(doc.stemmed) for doc in self.documents]
        X = self.vectorizer.fit_transform(docs)
        X = X.T.toarray()
        df = pd.DataFrame(X, index=self.vectorizer.get_feature_names())
        return df

    def query_vectorizer(self, query: Query) -> np.ndarray:
        """Return a frequency vector of a query over the member corpus vocabulary."""
        q = [query.stem()]
        q_vec = self.vectorizer.transform(q).toarray().reshape(self.vocab_len,)
        return q_vec

    def retrieve(self, query: Query, topk: int = 3, quiet: bool = True) -> Corpus:
        """Return topk candidates from the corpus most similar to the query."""

        q_vec = self.query_vectorizer(query)
        inverted_doc = self.inverted_doc.to_numpy()

        def similarity(q_vec, inverted_doc, size):
            sim = {}
            for i in range(size):
                sim[i] = (inverted_doc[:, i].dot(q_vec)) \
                    / np.linalg.norm(inverted_doc[:, i]) * \
                    np.linalg.norm(q_vec)
            return sim

        sim = similarity(q_vec, inverted_doc, self.doc_len)
        sim_sorted = sorted(sim.items(), key=lambda x: x[1], reverse=True)

        candidates = []
        for i, vals in enumerate(sim_sorted):
            if i >= topk:
                break

            k, v = vals
            if not quiet:
                print(f'Similarity: {v}')
                print(f'With Document: {doc.book}, {doc.chapter_no}')

            doc = self.documents[k]
            candidates.append(doc)

        return candidates


class PassageRetrieval(DocumentRetrieval):
    """Subclass of DocumentRetrieval with added functionality for recording passages."""
    def __init__(self, documents: Corpus) -> None:
        docs = self.get_passages(documents)
        super().__init__(docs)

    def get_passages(self, documents: Corpus) -> Corpus:
        """Returns a list of passages by chunking the candidate documents."""

        def chunks(doc: Document, threshold: int) -> str:
            text = str(doc)
            i = 0
            words = text.split(' ')
            while i < len(words):
                yield " ".join(words[i:i + threshold])
                i = i + threshold
            yield " ".join(words[i:])

        docs = []

        for doc in documents:
            for chunk in chunks(doc, PASSAGE_LEN):
                if chunk not in ["", " "]:
                    text = [x + '.' for x in chunk.split('.')]
                    docs.append(Document(doc.book, doc.chapter_no, text))

        return docs
