from rouge_score import rouge_scorer

'''
    Define your reference and generated summaries
    Replace with generated summaries
'''
reference_summary = "The quick brown fox jumps over the lazy dog."
generated_summary = "The fast brown fox leaps over the sleepy dog."

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Compute ROUGE scores
scores = scorer.score(reference_summary, generated_summary)

# Print ROUGE scores
print(f"ROUGE-1: Precision={scores['rouge1'].precision:.4f}, Recall={scores['rouge1'].recall:.4f}, F1 Score={scores['rouge1'].fmeasure:.4f}")
print(f"ROUGE-2: Precision={scores['rouge2'].precision:.4f}, Recall={scores['rouge2'].recall:.4f}, F1 Score={scores['rouge2'].fmeasure:.4f}")
print(f"ROUGE-L: Precision={scores['rougeL'].precision:.4f}, Recall={scores['rougeL'].recall:.4f}, F1 Score={scores['rougeL'].fmeasure:.4f}")
