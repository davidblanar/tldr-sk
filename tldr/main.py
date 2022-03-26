import simplemma
import re
import argparse
import numpy as np

REGEX = re.compile('[,.„“;/_()]')
LANG_DATA = simplemma.load_data('sk')


def process_sentence(sentence):
    words = re.sub(REGEX, '', sentence).split()
    return [simplemma.lemmatize(word, LANG_DATA) for word in words]


def calculate_word_score(sentences):
    score = {}
    for sentence in sentences:
        for word in sentence:
            if word in score:
                score[word] += 1
            else:
                score[word] = 1
    return score


def calculate_sentence_score(sentences, word_score, stop_words):
    scores = []
    for i, sentence in enumerate(sentences):
        if len(sentence) == 0:
            scores.append(0)
            continue
        score = 0
        for word in sentence:
            if word not in stop_words:
                score += word_score[word]
        scores.append(score)
    return scores


def count_words(sentences):
    return sum([len(sentence) for sentence in sentences])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=False, default='./tldr/articles/article.txt', help='Path to file to TLDR')
    parser.add_argument('--n-best', type=int, required=False, default=5, help='Number of best sentences to return')
    args = parser.parse_args()
    n = args.n_best
    file_path = args.file

    with open('./tldr/slovak_stopwords.txt', 'r') as f:
        stop_words = set(f.read().splitlines())

    with open(file_path, 'r') as f:
        sentences = [item for item in f.read().split('.') if item.strip() != '']
        processed = [process_sentence(sentence) for sentence in sentences]
        word_score = calculate_word_score(processed)
        sentence_score = calculate_sentence_score(processed, word_score, stop_words)
        best_indices = np.argsort(sentence_score)[::-1][:n]
        best_sentences = np.array(sentences)[sorted(best_indices)]
        reduction = round((1 - count_words(best_sentences) / count_words(sentences)) * 100)
        print(f"TLDR: (reduced by {reduction}%)")
        print(". ".join(best_sentences))
