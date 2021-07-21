import textract
import shutil
import gensim
from typing import Iterable, Tuple
import sys
from gensim.models import LdaModel
from gensim.corpora.textcorpus import TextDirectoryCorpus, TextCorpus
import os


MAX_WORD_FREQUENCY = 0.9
NUMBER_OF_TOPICS = 50


def extract_file_content(input_path: str, output_dir: str) -> str:
    """
    Returns the content as text and writes it to a file.
    May fail with UnicodeDecodeError when there is an unknown character in the file.
    """
    content = textract.process(input_path, 'utf8', method='pdfminer').decode('utf8')
    filename = os.path.basename(input_path)
    filename_without_ext = os.path.splitext(filename)[0]
    with open(os.path.join(output_dir, filename_without_ext) + '.txt', 'w') as file:
        file.write(content)
    return content


def extract_folder_content(input_path: str, output_path: str) -> Iterable[Tuple[str, str]]:
    """
    Extracts content from the .pdf files in the given folder and
    returns tuples where the first element is the filename and the second
    the content.
    """
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(".pdf"):
                full_path = os.path.join(root, file)
                try:
                    yield (full_path, extract_file_content(full_path, output_path))
                except UnicodeDecodeError:
                    print('Unable to extract text from ' + full_path)


def tokenize(text: str) -> Iterable[str]:
    tokens = list(gensim.utils.tokenize(text, lowercase=True, deacc=True, errors='ignore'))
    bigrams = [f'{t1} {t2}' for t1, t2 in zip(tokens[:-1], tokens[1:])]
    return tokens + bigrams


class TxtCorpus(TextCorpus):
    def __init__(self, path, dictionary):
        self._path = path
        super().__init__(path, dictionary=dictionary, tokenizer=tokenize)

    def getstream(self):
        with open(self._path, errors='ignore') as f:
            yield f.read()


if __name__ == '__main__':
    training_folder = sys.argv[1]
    input_folder = sys.argv[2]
    try:
        shutil.rmtree('output')
    except FileNotFoundError:
        pass
    os.mkdir('output')
    list(extract_folder_content(training_folder, 'output'))
    corpus = TextDirectoryCorpus('output', tokenizer=tokenize)
    corpus.dictionary.filter_extremes(no_above=MAX_WORD_FREQUENCY)
    corpus = TextDirectoryCorpus('output', tokenizer=tokenize, dictionary=corpus.dictionary)
    model = LdaModel(corpus, num_topics=NUMBER_OF_TOPICS)
    for name, content in extract_folder_content(input_folder, 'output'):
        with open('output/topics.txt', 'a') as topics_file:
            topics_file.write(name + '\n')
            for topic_distr in model[TxtCorpus(name, corpus.dictionary)]:
                for topic_id, weight in topic_distr:
                    words = [(corpus.dictionary[word_id], word_weight) for word_id, word_weight in model.get_topic_terms(topic_id, 20)]
                    topics_file.write(f'{str(weight)} {str(words)}\n')
            topics_file.write('\n')
