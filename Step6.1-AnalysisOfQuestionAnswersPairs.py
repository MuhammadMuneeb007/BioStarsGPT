import json
import time
import os
from tqdm import tqdm
import numpy as np
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, LongformerTokenizer, LongformerForSequenceClassification
import torch.nn.functional as F

def load_json_file(file_path):
    """Load JSON file and return the data"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def setup_qa_evaluation_model(model_choice="llama3.3"):
    """Set up the question-answering evaluation model from Hugging Face
    
    Args:
        model_choice: String indicating which model to use:
            - "longformer": Allenai's Longformer for handling long sequences
            - "deberta": Microsoft's DeBERTa which has strong performance on NLI tasks
            - "llama33": Meta's Llama 3.3 model (fine-tuned for NLI tasks)
            - "original": The original model from the script (QNLI-ELECTRA)
    """
    # No try-except blocks - let errors propagate upward
    if model_choice == "longformer":
        # Longformer can handle sequences up to 4096 tokens
        model_name = "allenai/longformer-large-4096-finetuned-triviaqa"
        tokenizer = LongformerTokenizer.from_pretrained(model_name)
        model = LongformerForSequenceClassification.from_pretrained(model_name, num_labels=2)
        max_length = 4096
        print("Using Longformer model with 4096 token support")
        
    elif model_choice == "deberta":
        # DeBERTa has strong performance on NLI tasks
        model_name = "microsoft/deberta-v3-large"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
        max_length = 2048
        print("Using DeBERTa model with 2048 token support")
        
    elif model_choice == "roberta-mnli":
        # RoBERTa fine-tuned on MNLI for natural language inference
        model_name = "roberta-large-mnli"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        max_length = 512
        print("Using RoBERTa-MNLI model with 512 token support")
        
    elif model_choice == "llama3.3":
        # Llama 3.3 for NLI tasks
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
        max_length = 1024  # Llama 3.3 supports very long contexts
        print("Using Llama 3.3 model with 8192 token support")
        
    else:  # "original"
        # Original model from the script
        model_name = "cross-encoder/qnli-electra-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        max_length = 512
        print("Using original QNLI-ELECTRA model with 512 token support")
    
    return tokenizer, model, max_length

def chunk_and_process_long_texts(question, explanation, model, tokenizer, max_length):
    """Process long texts by chunking and scoring relevance of each chunk"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Check if this is the Llama 3.3 model (special handling for instruction-tuned models)
    is_llama33 = "llama" in str(model.__class__).lower() or "meta-llama" in str(getattr(model, "_name_or_path", "")).lower()
    
    # For the question, we'll keep it as is since it's typically shorter
    # For the explanation, we'll chunk it if needed
    
    # Tokenize the question to get its length
    question_tokens = tokenizer.encode(question, add_special_tokens=False)
    
    # Reserve tokens for special tokens and the question
    # Using a conservative estimate: [CLS] question [SEP] explanation [SEP]
    reserved_tokens = len(question_tokens) + 3
    available_tokens = max_length - reserved_tokens
    
    if available_tokens <= 0:
        # Question itself is too long, truncate it
        truncated_question = tokenizer.decode(question_tokens[:max_length-4], skip_special_tokens=True)
        question = truncated_question
        available_tokens = max_length - len(tokenizer.encode(question, add_special_tokens=False)) - 3
    
    # Tokenize the explanation
    explanation_tokens = tokenizer.encode(explanation, add_special_tokens=False)
    
    # If the explanation fits within available tokens, process as is
    if len(explanation_tokens) <= available_tokens:
        # Special handling for instruction-tuned models like Llama 3.3
        if is_llama33:
            prompt = f"<s>[INST] Given the question: '{question}', determine if the following text answers this question.\n\nText: {explanation}\n\nDoes this text provide a direct answer to the question? Respond with 'entailment' if yes, 'neutral' if partially, and 'contradiction' if not at all. [/INST]"
        else:
            prompt = f"Question: {question} Does the following text answer this question? {explanation}"
        
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            if is_llama33:
                # For instruction models like Llama, analyze logits differently
                # We map to entailment, neutral, contradiction classification
                entail_token_ids = tokenizer.encode("entailment", add_special_tokens=False)
                neutral_token_ids = tokenizer.encode("neutral", add_special_tokens=False) 
                contra_token_ids = tokenizer.encode("contradiction", add_special_tokens=False)
                
                # Get logits for the first token of each class
                entail_logit = logits[0, -1, entail_token_ids[0]]
                neutral_logit = logits[0, -1, neutral_token_ids[0]]
                contra_logit = logits[0, -1, contra_token_ids[0]]
                
                # Create pseudo-probabilities using softmax
                class_logits = torch.tensor([contra_logit, neutral_logit, entail_logit]).unsqueeze(0)
                probs = F.softmax(class_logits, dim=1)
                
                # Entailment is the last position (2)
                score = probs[0][2].item() * 100
                is_answered = score >= 40  # Lower threshold for generative models
            else:
                # Standard NLI models
                probs = torch.softmax(logits, dim=1)
                
                if probs.shape[1] == 2:  # Binary classification (QNLI)
                    score = probs[0][1].item() * 100  # Entailment probability as percentage
                    is_answered = score >= 50  # Threshold
                elif probs.shape[1] == 3:  # 3-class (MNLI)
                    score = probs[0][2].item() * 100  # Entailment probability as percentage
                    is_answered = score >= 40  # Lower threshold for 3-class model
                else:
                    # Default fallback
                    score = probs[0][-1].item() * 100
                    is_answered = score >= 50
                
        return {
            "is_answered": bool(is_answered),
            "score": float(score),
            "confidence": float(score) / 100,
            "method": "full_text"
        }
    
    # If the explanation is too long, chunk it and process each chunk
    else:
        # Chunk the explanation into overlapping segments
        chunk_size = available_tokens
        stride = chunk_size // 2  # 50% overlap between chunks
        
        chunks = []
        for i in range(0, len(explanation_tokens), stride):
            chunk = explanation_tokens[i:i+chunk_size]
            if len(chunk) < chunk_size // 4:  # Skip very small final chunks
                continue
            chunks.append(tokenizer.decode(chunk, skip_special_tokens=True))
        
        # Process each chunk
        chunk_scores = []
        for chunk in chunks:
            prompt = f"Question: {question} Does the following text answer this question? {chunk}"
            
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                
                if probs.shape[1] == 2:  # Binary classification (QNLI)
                    score = probs[0][1].item() * 100  # Entailment probability as percentage
                elif probs.shape[1] == 3:  # 3-class (MNLI)
                    score = probs[0][2].item() * 100  # Entailment probability as percentage
                else:
                    # Default fallback
                    score = probs[0][-1].item() * 100
                    
                chunk_scores.append(score)
        
        # Aggregate chunk scores
        # We'll use the maximum score, as we're interested if ANY part of the 
        # explanation answers the question
        max_score = max(chunk_scores) if chunk_scores else 0
        avg_score = sum(chunk_scores) / len(chunk_scores) if chunk_scores else 0
        
        # Determine if the question is answered based on max score
        # Need to get model output classes correctly - this is a likely error point
        if hasattr(model.config, 'num_labels') and model.config.num_labels == 3:  # 3-class (MNLI)
            is_answered = max_score >= 40  # Lower threshold for 3-class model
        else:
            is_answered = max_score >= 50
            
        return {
            "is_answered": bool(is_answered),
            "score": float(max_score),
            "avg_score": float(avg_score),
            "max_score": float(max_score),
            "confidence": float(max_score) / 100,
            "chunks_analyzed": len(chunk_scores),
            "method": "chunked"
        }

def count_question_explanation_pairs(data):
    """Count the number of entries with Question1/Explanation1 and Question2/Explanation2 pairs"""
    q1_e1_count = 0
    q2_e2_count = 0
    
    for entry in data:
        has_q1_e1 = "Question1" in entry and entry["Question1"] and "Explanation1" in entry and entry["Explanation1"]
        has_q2_e2 = "Question2" in entry and entry["Question2"] and "Explanation2" in entry and entry["Explanation2"]
        
        if has_q1_e1:
            q1_e1_count += 1
        if has_q2_e2:
            q2_e2_count += 1
    
    return {
        "Question1_Explanation1_pairs": q1_e1_count,
        "Question2_Explanation2_pairs": q2_e2_count,
        "Total_entries": len(data)
    }

def analyze_technical_question_answering(data, model, tokenizer, max_length, batch_size=1, sample_size=None):
    """Analyze if technical questions are answered in the explanations"""
    
    # Sample the data if requested
    if sample_size and sample_size < len(data):
        import random
        sampled_data = random.sample(data, sample_size)
        print(f"Sampled {sample_size} entries from {len(data)} total entries")
    else:
        sampled_data = data
    
    # First, count question-explanation pairs
    pair_counts = count_question_explanation_pairs(sampled_data)
    print("\n=== Question-Explanation Pair Statistics ===")
    print(f"Total entries: {pair_counts['Total_entries']}")
    print(f"Entries with Question1/Explanation1 pairs: {pair_counts['Question1_Explanation1_pairs']} ({pair_counts['Question1_Explanation1_pairs']/pair_counts['Total_entries']*100:.2f}%)")
    print(f"Entries with Question2/Explanation2 pairs: {pair_counts['Question2_Explanation2_pairs']} ({pair_counts['Question2_Explanation2_pairs']/pair_counts['Total_entries']*100:.2f}%)")
    
    # Prepare to store results
    q1_e1_scores = []
    q2_e2_scores = []
    q1_e1_answered = 0
    q2_e2_answered = 0
    chunk_stats = {"total_chunks": 0, "entries_chunked": 0}
    analyzed_count = 0
    
    # Process data
    for i in tqdm(range(len(sampled_data)), desc="Processing entries"):
        entry = sampled_data[i]
        
        # Debug output to verify entry structure
        if i == 0:
            print(f"Sample entry keys: {list(entry.keys())}")
        
        # Initialize question_answering_analysis if it doesn't exist
        if "question_answering_analysis" not in entry:
            entry["question_answering_analysis"] = {
                "explanations_analysis": {},
                "processing_info": {"max_token_length": max_length}
            }
        
        analyzed_this_entry = False
        
        # First process Question1/Explanation1 if available
        has_q1_e1 = "Question1" in entry and entry["Question1"] and "Explanation1" in entry and entry["Explanation1"]
        
        if has_q1_e1:
            analyzed_this_entry = True
            
            # Debug first question-explanation pair
            if i == 0:
                print(f"Sample Question1: {entry['Question1'][:100]}...")
                print(f"Sample Explanation1: {entry['Explanation1'][:100]}...")
            
            # Process question and explanation
            result = chunk_and_process_long_texts(
                entry["Question1"], 
                entry["Explanation1"], 
                model, 
                tokenizer, 
                max_length
            )
            
            # Store the analysis
            entry["question_answering_analysis"]["explanations_analysis"]["Question1_Explanation1"] = result
            
            # Update statistics
            q1_e1_scores.append(result["score"])
            if result["is_answered"]:
                q1_e1_answered += 1
            
            # Track chunking statistics
            if "method" in result and result["method"] == "chunked":
                chunk_stats["entries_chunked"] += 1
                if "chunks_analyzed" in result:
                    chunk_stats["total_chunks"] += result["chunks_analyzed"]
        
        # Then process Question2/Explanation2 if available
        has_q2_e2 = "Question2" in entry and entry["Question2"] and "Explanation2" in entry and entry["Explanation2"]
        
        if has_q2_e2:
            analyzed_this_entry = True
            
            # Debug first question-explanation pair
            if i == 0:
                print(f"Sample Question2: {entry['Question2'][:100]}...")
                print(f"Sample Explanation2: {entry['Explanation2'][:100]}...")
            
            # Process question and explanation
            result = chunk_and_process_long_texts(
                entry["Question2"], 
                entry["Explanation2"], 
                model, 
                tokenizer, 
                max_length
            )
            
            # Store the analysis
            entry["question_answering_analysis"]["explanations_analysis"]["Question2_Explanation2"] = result
            
            # Update statistics
            q2_e2_scores.append(result["score"])
            if result["is_answered"]:
                q2_e2_answered += 1
            
            # Track chunking statistics
            if "method" in result and result["method"] == "chunked":
                chunk_stats["entries_chunked"] += 1
                if "chunks_analyzed" in result:
                    chunk_stats["total_chunks"] += result["chunks_analyzed"]
        
        if analyzed_this_entry:
            analyzed_count += 1
    
    # Calculate overall statistics
    total_analyzed = q1_e1_answered + q2_e2_answered
    total_questions = len(q1_e1_scores) + len(q2_e2_scores)
    
    # Count entries where at least one question was answered
    entries_with_answered_questions = 0
    for entry in sampled_data:
        if "question_answering_analysis" in entry:
            explanations = entry["question_answering_analysis"].get("explanations_analysis", {})
            if any(exp_data.get("is_answered", False) for exp_key, exp_data in explanations.items()):
                entries_with_answered_questions += 1
    
    # Compile statistics
    stats = {
        "total_entries": len(data),
        "analyzed_entries": analyzed_count,
        "question_explanation_pairs": pair_counts,
        "question1_explanation1": {
            "total": len(q1_e1_scores),
            "answered": q1_e1_answered,
            "percentage_answered": (q1_e1_answered / len(q1_e1_scores) * 100) if q1_e1_scores else 0,
            "avg_score": np.mean(q1_e1_scores) if q1_e1_scores else 0,
            "median_score": np.median(q1_e1_scores) if q1_e1_scores else 0
        },
        "question2_explanation2": {
            "total": len(q2_e2_scores),
            "answered": q2_e2_answered,
            "percentage_answered": (q2_e2_answered / len(q2_e2_scores) * 100) if q2_e2_scores else 0,
            "avg_score": np.mean(q2_e2_scores) if q2_e2_scores else 0,
            "median_score": np.median(q2_e2_scores) if q2_e2_scores else 0
        },
        "combined_statistics": {
            "total_questions": total_questions,
            "total_answered": total_analyzed,
            "percentage_answered": (total_analyzed / total_questions * 100) if total_questions > 0 else 0,
            "entries_with_at_least_one_answered": entries_with_answered_questions,
            "percentage_entries_with_answered": (entries_with_answered_questions / analyzed_count * 100) if analyzed_count > 0 else 0
        },
        "chunking_stats": {
            "entries_requiring_chunking": chunk_stats["entries_chunked"],
            "percentage_chunked": (chunk_stats["entries_chunked"] / analyzed_count * 100) if analyzed_count > 0 else 0,
            "total_chunks_processed": chunk_stats["total_chunks"],
            "avg_chunks_per_chunked_entry": (chunk_stats["total_chunks"] / chunk_stats["entries_chunked"]) 
                                           if chunk_stats["entries_chunked"] > 0 else 0
        },
        "model_info": {
            "max_token_length": max_length
        }
    }
    
    return sampled_data, stats

def main():
    # Load data
    input_path = "filtered_questions_intermediate.json"
    print(f"Loading data from {input_path}...")
    data = load_json_file(input_path)
    
    print(f"Total entries loaded: {len(data)}")
    
    # Ask for model choice
    print("\nAvailable models:")
    print("1. Longformer (supports up to 4096 tokens)")
    print("2. DeBERTa (supports up to 2048 tokens)")
    print("3. RoBERTa-MNLI (supports up to 512 tokens)")
    print("4. Llama 3.3 (supports up to 8192 tokens)")
    print("5. Original QNLI-ELECTRA (supports up to 512 tokens)")
    
    model_choice = 1
    #model_choice = int(input("Select model (1-5): "))
    
    if model_choice == 1:
        model_name = "longformer"
    elif model_choice == 2:
        model_name = "deberta"
    elif model_choice == 3:
        model_name = "roberta-mnli"
    elif model_choice == 4:
        model_name = "llama3.3"
    else:
        model_name = "original"
    
    # Ask for sample size
    sample_size = 10000
    sample_size = sample_size if sample_size > 0 else None
    
    # Set up the model
    print("Loading selected model...")
    tokenizer, model, max_length = setup_qa_evaluation_model(model_choice=model_name)
    print(f"Model loaded successfully with {max_length} token support")
    
    # Analyze technical question-answer alignment
    print(f"Starting analysis with {max_length} token limit...")
    processed_data, stats = analyze_technical_question_answering(
        data, 
        model, 
        tokenizer,
        max_length,
        sample_size=sample_size
    )
    
    # Save results
    output_path = f"filtered_questions_with_{model_name}_analysis.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    # Save statistics separately
    stats_output_path = f"{model_name}_question_answering_statistics.json"
    with open(stats_output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # Print summary statistics
    print("\n=== Technical Question Answering Analysis ===")
    print(f"Model used: {model_name} with {max_length} token limit")
    print(f"Total entries analyzed: {stats['analyzed_entries']}")
    
    print("\nQuestion1/Explanation1 Statistics:")
    print(f"  Total pairs: {stats['question1_explanation1']['total']}")
    print(f"  Questions answered: {stats['question1_explanation1']['answered']} ({stats['question1_explanation1']['percentage_answered']:.2f}%)")
    print(f"  Average score: {stats['question1_explanation1']['avg_score']:.2f}/100")
    print(f"  Median score: {stats['question1_explanation1']['median_score']:.2f}/100")
    
    print("\nQuestion2/Explanation2 Statistics:")
    print(f"  Total pairs: {stats['question2_explanation2']['total']}")
    print(f"  Questions answered: {stats['question2_explanation2']['answered']} ({stats['question2_explanation2']['percentage_answered']:.2f}%)")
    print(f"  Average score: {stats['question2_explanation2']['avg_score']:.2f}/100")
    print(f"  Median score: {stats['question2_explanation2']['median_score']:.2f}/100")
    
    print("\nCombined Statistics:")
    print(f"  Total questions analyzed: {stats['combined_statistics']['total_questions']}")
    print(f"  Total questions answered: {stats['combined_statistics']['total_answered']} ({stats['combined_statistics']['percentage_answered']:.2f}%)")
    print(f"  Entries with at least one answered question: {stats['combined_statistics']['entries_with_at_least_one_answered']} ({stats['combined_statistics']['percentage_entries_with_answered']:.2f}%)")
    
    print("\nChunking statistics:")
    print(f"  Entries requiring chunking: {stats['chunking_stats']['entries_requiring_chunking']} ({stats['chunking_stats']['percentage_chunked']:.2f}%)")
    print(f"  Total chunks processed: {stats['chunking_stats']['total_chunks_processed']}")
    if stats['chunking_stats']['entries_requiring_chunking'] > 0:
        print(f"  Average chunks per chunked entry: {stats['chunking_stats']['avg_chunks_per_chunked_entry']:.2f}")
    
    print(f"\nResults saved to {output_path}")
    print(f"Statistics saved to {stats_output_path}")

if __name__ == "__main__":
    main()