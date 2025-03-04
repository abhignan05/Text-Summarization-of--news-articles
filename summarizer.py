import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter

nltk.download('punkt')

def generate_summary(text):
    sentences = sent_tokenize(text)
    
    if len(sentences) <= 3:
        return text  # Return the full text if it's short

    # Word Frequency-Based Summarization (Extractive Approach)
    words = word_tokenize(text.lower())
    word_freq = Counter(words)

    sentence_scores = {}
    for sentence in sentences:
        score = sum(word_freq[word.lower()] for word in word_tokenize(sentence) if word.lower() in word_freq)
        sentence_scores[sentence] = score

    # Select top 30% of sentences based on scores
    num_sentences = max(1, len(sentences) // 3)
    summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    
    summary = " ".join(summary_sentences)
    return summary
