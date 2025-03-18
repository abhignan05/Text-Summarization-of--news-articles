from transformers import pipeline
import torch
import gc  # Garbage collection

# Optimized lightweight model
model_name = "sshleifer/distilbart-cnn-12-6"
summarizer = pipeline("summarization", model=model_name, device=-1)  # CPU deployment

def summarize_text(text):
    """
    Summarizes input text while ensuring complete, structured sentences.
    """
    
    # Merge paragraphs into a single string
    cleaned_text = " ".join(text.split("\n"))

    # Summary constraints to prevent sentence truncation
    max_len = 160  # Ensure enough space for full sentences
    min_len = 80   # Prevent very short, incomplete summaries

    # Generate summary
    summary = summarizer(
        cleaned_text, 
        max_length=max_len, 
        min_length=min_len, 
        length_penalty=2.0,  # Promotes sentence structure
        early_stopping=True,  # Stops generating mid-word
        clean_up_tokenization_spaces=True,  # Removes unnecessary spaces
        do_sample=False
    )

    # Clear memory after processing
    gc.collect()

    return summary[0]['summary_text']
