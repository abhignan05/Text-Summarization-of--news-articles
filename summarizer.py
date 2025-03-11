from transformers import pipeline

# Load the summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def generate_summary(text):
    """
    Generate an abstractive summary of the given text.
    
    - If the text is long (more than 2 paragraphs), return 4-5 lines.
    - If the text is short (1-2 paragraphs), return 2-3 lines.
    """
    word_count = len(text.split())  # Count words in input text

    if word_count > 100:  # If the article is long (more than ~2 paragraphs)
        min_len = 80   # Ensure at least 4-5 lines
        max_len = 120
    else:  # If the article is short (1-2 paragraphs)
        min_len = 40   # Ensure at least 2-3 lines
        max_len = 60

    # Generate summary
    summary = summarizer(text, max_length=max_len, min_length=min_len, length_penalty=1.0, do_sample=False)
    return summary[0]['summary_text']
