import math
import string

import nltk
import sys
import os

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():
    # # Check command-line arguments
    # if len(sys.argv) != 2:
    #     sys.exit("Usage: python questions.py corpus")
    #
    # # Calculate IDF values across files
    # files = load_files(sys.argv[1])
    files = load_files("corpus")
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = dict()
    for file in os.listdir(directory):
        with open(os.path.join(directory, file), "r", encoding="utf-8") as f:
            files[file] = f.read()
    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    words = nltk.word_tokenize(document.lower())
    words = [word for word in words
             if word not in nltk.corpus.stopwords.words("english")
             and word not in string.punctuation]
    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idfs = dict()
    words = set()
    for doc in documents:
        words = words.union(set(documents[doc]))
    for word in words:
        val = 0
        for doc in documents:
            if word in documents[doc]:
                val += 1
        idfs[word] = math.log(len(documents.keys()) / val)
    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the  `n` top
    files that match the query, ranked according to tf-idf.
    """
    query_files = {key: 0 for key in files.keys()}
    for word in query:
        idf = idfs[word]
        for name, content in files.items():
            tf_idf = content.count(word) / len(files) * idf
            query_files[name] += tf_idf
    docs = [i[0] for i in sorted(query_files.items(), key=lambda s: s[1], reverse=True)][0:n]
    return docs


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    query_sens = {key: [0, 0] for key in sentences.keys()}
    for name in sentences:
        density = 0
        for word in query:
            if word in sentences[name]:
                density += 1
            query_sens[name][0] += idfs[word]
        query_sens[name][1] = density/len(sentences)
    docs = [i[0] for i in sorted(query_sens.items(), key=lambda s: (s[1][0],s[1][1]), reverse=True)][0:n]
    return docs


if __name__ == "__main__":
    main()
