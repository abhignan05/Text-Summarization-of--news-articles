import nltk
import torch
from transformers import pipeline
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

# Load Hugging Face Abstractive Model
abstractive_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def extractive_summary(text, num_sentences=3):
    """Extractive summarization using sentence ranking (basic approach)."""
    sentences = sent_tokenize(text)
    return ' '.join(sentences[:num_sentences]) if len(sentences) > num_sentences else text

def abstractive_summary(text):
    """Abstractive summarization using Transformer model."""
    return abstractive_summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']

def generate_summary(text, summary_type):
    """Decides which summarization method to use."""
    if summary_type == "extractive":
        return extractive_summary(text)
    elif summary_type == "abstractive":
        return abstractive_summary(text)
    return "Invalid summarization type selected."
