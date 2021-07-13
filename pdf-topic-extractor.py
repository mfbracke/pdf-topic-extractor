import textract
import shutil
from typing import Iterable, Tuple
import sys
from gensim.models import LdaModel
from gensim.corpora.textcorpus import TextDirectoryCorpus, TextCorpus
import os


NUMBER_OF_TOPICS = 10


def extract_file_content(input_path: str, output_dir: str) -> str:
    """
    Returns the content as text and writes it to a file.
    May fail with UnicodeDecodeError when there is an unknown character in the file.
    """
    content = textract.process(input_path, 'utf8').decode('utf8')
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

class TxtCorpus(TextCorpus):
    def __init__(self, path, dictionary):
        self._path = path
        super().__init__(path, dictionary=dictionary)

    def getstream(self):
        with open(self._path, errors='ignore') as f:
            yield f.read()


if __name__ == '__main__':
    input_folder = sys.argv[1]
    try:
        shutil.rmtree('output')
    except FileNotFoundError:
        pass
    os.mkdir('output')
    contents_and_names = list(extract_folder_content(input_folder, 'output'))
    corpus = TextDirectoryCorpus('output')
    model = LdaModel(corpus, num_topics=NUMBER_OF_TOPICS)
    for name, content in contents_and_names:
        with open('output/topics.txt', 'a') as topics_file:
            topics_file.write(name + '\n')
            for topic_distr in model[TxtCorpus(name, corpus.dictionary)]:
                for topic_id, weight in topic_distr:
                    words = [(corpus.dictionary[word_id], word_weight) for word_id, word_weight in model.get_topic_terms(topic_id, 20)]
                    topics_file.write(f'{str(weight)} {str(words)}\n')
            topics_file.write('\n')
