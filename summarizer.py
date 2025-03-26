from transformers import pipeline
import torch
import gc  # Garbage collection

# Optimized lightweight model
model_name = "sshleifer/distilbart-cnn-12-6"
summarizer = pipeline("summarization", model=model_name, device=-1)

def summarize_text(text):
    """
    Dynamically adjusts summary length based on input size.
    """

    # Clean input text (merge paragraphs)
    cleaned_text = " ".join(text.split("\n"))

    # Count words to determine summary length
    word_count = len(cleaned_text.split())

    # Set dynamic length constraints
    if word_count <= 100:  # Short text (1 para)
        min_len, max_len = 40, 60
    elif word_count <= 250:  # Medium text (2-3 paras)
        min_len, max_len = 80, 120
    else:  # Long text (4+ paras)
        min_len, max_len = 130, 200

    # Generate summary
    summary = summarizer(
        cleaned_text,
        max_length=max_len,
        min_length=min_len,
        length_penalty=1.8,  # Encourage more complete output
        early_stopping=True,
        clean_up_tokenization_spaces=True,
        do_sample=False
    )

    # Clean memory
    gc.collect()

    return summary[0]['summary_text']
