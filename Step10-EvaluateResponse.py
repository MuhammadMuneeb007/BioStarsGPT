import json
import os
import glob
import re
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from prettytable import PrettyTable
import spacy
import pandas as pd
from difflib import SequenceMatcher
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from collections import defaultdict
import Levenshtein
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import warnings
import nltk
os.environ['GENSIM_DATA_DIR'] = '/data/ascher02/uqmmune1/BioStarsGPT/temp/'
os.environ['TRANSFORMERS_CACHE'] = '/data/ascher02/uqmmune1/BioStarsGPT/temp/huggingface'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/data/ascher02/uqmmune1/BioStarsGPT/temp/sentence_transformers'
os.environ["TRITON_CACHE_DIR"] = "/data/ascher02/uqmmune1/BioStarsGPT/temp/.triton_cache"
os.environ["HF_HOME"] = "/data/ascher02/uqmmune1/BioStarsGPT/temp/.huggingface_cache"
os.environ["TORCH_HOME"] = "/data/ascher02/uqmmune1/BioStarsGPT/temp/.torch_cache"

# Set NLTK to use your current project path or a subfolder
nltk_data_dir = '/data/ascher02/uqmmune1/BioStarsGPT/temp/nltk_data'
nltk.data.path.append(nltk_data_dir)

# Download necessary corpora directly there
nltk.download('wordnet', download_dir=nltk_data_dir)
nltk.download('omw-1.4', download_dir=nltk_data_dir)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

print("Loading required resources...")

# Ensure NLTK resources are available
for resource in ['punkt', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

# Load spaCy model
nlp = spacy.load("en_core_web_md")

# Initialize ROUGE scorer
rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Check for BERT model availability
BERT_AVAILABLE = False
try:
    print("Loading sentence-transformer model...")
    from sentence_transformers import SentenceTransformer
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    BERT_AVAILABLE = True
    print("Sentence transformer model loaded!")
except:
    print("Could not load sentence-transformer model. Some metrics will be unavailable.")

# Try to load NLI model for entailment
NLI_AVAILABLE = False
try:
    print("Loading NLI model for entailment detection...")
    nli_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    NLI_AVAILABLE = True
    print("NLI model loaded!")
except:
    print("Could not load NLI model. Entailment metrics will be unavailable.")

# Try to load gensim for WMD
WMD_AVAILABLE = False
try:
    print("Loading Gensim for Word Mover's Distance...")
    import gensim.downloader as api
    word_vectors = api.load("glove-wiki-gigaword-100")
    WMD_AVAILABLE = True
    print("Word vectors loaded for WMD!")
except:
    print("Could not load Gensim or word vectors. WMD will be unavailable.")


def read_raw_responses(file_path):
    """Read all responses from the jsonl file."""
    responses = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            response = json.loads(line.strip())
            responses.append(response)
    return responses


def calculate_exact_match(references, predictions):
    """Calculate exact match percentage."""
    exact_matches = [1 if ref == pred else 0 for ref, pred in zip(references, predictions)]
    return np.mean(exact_matches)


def calculate_levenshtein_similarity(references, predictions):
    """Calculate normalized Levenshtein similarity (1 - distance/max_length)."""
    similarities = []
    for ref, pred in zip(references, predictions):
        # Limit text length to avoid excessive processing
        ref = ref[:10000]
        pred = pred[:10000]
        max_len = max(len(ref), len(pred))
        if max_len == 0:  # Both strings are empty
            similarities.append(1.0)
        else:
            distance = Levenshtein.distance(ref, pred)
            # Normalize to a similarity score between 0 and 1
            similarity = 1 - (distance / max_len)
            similarities.append(similarity)
    return np.mean(similarities)


def calculate_jaccard_similarity(references, predictions):
    """Calculate Jaccard similarity between tokenized texts."""
    similarities = []
    for ref, pred in zip(references, predictions):
        # Tokenize and create sets
        ref_tokens = set(word_tokenize(ref.lower()))
        pred_tokens = set(word_tokenize(pred.lower()))
        
        # Calculate Jaccard similarity
        if not ref_tokens and not pred_tokens:  # Both are empty
            similarities.append(1.0)
        else:
            intersection = len(ref_tokens.intersection(pred_tokens))
            union = len(ref_tokens.union(pred_tokens))
            similarities.append(intersection / union)
    
    return np.mean(similarities)


def calculate_tfidf_cosine_similarity(references, predictions):
    """Calculate average cosine similarity using TF-IDF vectors."""
    vectorizer = TfidfVectorizer()
    
    # Combine references and predictions for vectorization
    all_texts = references + predictions
    
    # Create TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Split the matrix back into references and predictions
    references_tfidf = tfidf_matrix[:len(references)]
    predictions_tfidf = tfidf_matrix[len(references):]
    
    # Calculate cosine similarity for each pair
    similarities = []
    for i in range(len(references)):
        sim = cosine_similarity(references_tfidf[i], predictions_tfidf[i])[0][0]
        similarities.append(sim)
    
    return np.mean(similarities)


def calculate_rouge_scores(references, predictions):
    """Calculate ROUGE scores."""
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for ref, pred in zip(references, predictions):
        # Limit text length to avoid excessive processing
        ref = ref[:10000]
        pred = pred[:10000]
        
        # Calculate ROUGE scores
        scores = rouge_scorer.score(ref, pred)
        
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    return {
        'rouge1': np.mean(rouge1_scores),
        'rouge2': np.mean(rouge2_scores),
        'rougeL': np.mean(rougeL_scores)
    }


def calculate_bleu_score(references, predictions):
    """Calculate BLEU score."""
    smoothing = SmoothingFunction().method1
    references_tokenized = [[word_tokenize(ref.lower())] for ref in references]
    predictions_tokenized = [word_tokenize(pred.lower()) for pred in predictions]
    
    # Calculate BLEU scores
    bleu1 = corpus_bleu(references_tokenized, predictions_tokenized, 
                       weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu4 = corpus_bleu(references_tokenized, predictions_tokenized, 
                       smoothing_function=smoothing)
    
    return {
        'bleu1': bleu1,
        'bleu4': bleu4
    }


def calculate_meteor_score(references, predictions):
    """Calculate METEOR score."""
    scores = []
    for ref, pred in zip(references, predictions):
        ref_tokens = word_tokenize(ref.lower())
        pred_tokens = word_tokenize(pred.lower())
        score = meteor_score([ref_tokens], pred_tokens)
        scores.append(score)
    
    return np.mean(scores)


def calculate_spacy_similarity(references, predictions):
    """Calculate average semantic similarity using spaCy."""
    similarities = []
    for ref, pred in zip(references, predictions):
        # Process texts with spaCy
        ref_doc = nlp(ref[:10000])  # Limit length to prevent memory issues
        pred_doc = nlp(pred[:10000])
        
        # Calculate similarity
        similarity = ref_doc.similarity(pred_doc)
        similarities.append(similarity)
    
    return np.mean(similarities)


def calculate_sbert_similarity(references, predictions):
    """Calculate sentence-BERT similarity."""
    if not BERT_AVAILABLE:
        return 0.0
    
    # Encode sentences
    ref_embeddings = sentence_model.encode(references, convert_to_tensor=True)
    pred_embeddings = sentence_model.encode(predictions, convert_to_tensor=True)
    
    # Calculate cosine similarity
    similarities = []
    for i in range(len(references)):
        # Convert tensors to numpy for dot product calculation
        ref_emb = ref_embeddings[i].cpu().numpy()
        pred_emb = pred_embeddings[i].cpu().numpy()
        
        # Normalize embeddings
        ref_emb = ref_emb / np.linalg.norm(ref_emb)
        pred_emb = pred_emb / np.linalg.norm(pred_emb)
        
        # Calculate cosine similarity
        sim = np.dot(ref_emb, pred_emb)
        similarities.append(sim)
    
    return np.mean(similarities)


def calculate_word_movers_distance(references, predictions):
    """Calculate Word Mover's Distance and convert to similarity."""
    if not WMD_AVAILABLE:
        return 0.0
    
    similarities = []
    for ref, pred in zip(references, predictions):
        # Preprocess and tokenize
        ref_tokens = [w for w in word_tokenize(ref.lower()) if w in word_vectors]
        pred_tokens = [w for w in word_tokenize(pred.lower()) if w in word_vectors]
        
        if not ref_tokens or not pred_tokens:
            similarities.append(0.0)
            continue
        
        # Calculate WMD
        try:
            distance = word_vectors.wmdistance(ref_tokens, pred_tokens)
            # Convert distance to similarity (closer to 1 is better)
            # Using an exponential decay function
            similarity = np.exp(-distance)
            similarities.append(similarity)
        except:
            similarities.append(0.0)
    
    return np.mean(similarities) if similarities else 0.0


def calculate_entailment_score(references, predictions):
    """Calculate bidirectional entailment score using NLI model."""
    if not NLI_AVAILABLE:
        return 0.0
    
    scores = []
    for ref, pred in zip(references, predictions):
        # Skip if either text is too long
        if len(ref) > 1000 or len(pred) > 1000:
            scores.append(0.0)
            continue
        
        # Check if reference entails prediction
        ref_entails_pred = nli_model(pred, candidate_labels=["entailment", "contradiction"], 
                                    hypothesis_template="{}",
                                    multi_label=False)
        
        # Check if prediction entails reference
        pred_entails_ref = nli_model(ref, candidate_labels=["entailment", "contradiction"], 
                                    hypothesis_template="{}",
                                    multi_label=False)
        
        # Get probability scores for entailment
        ref_entails_pred_score = ref_entails_pred['scores'][ref_entails_pred['labels'].index("entailment")]
        pred_entails_ref_score = pred_entails_ref['scores'][pred_entails_ref['labels'].index("entailment")]
        
        # Average of bidirectional entailment
        avg_score = (ref_entails_pred_score + pred_entails_ref_score) / 2
        scores.append(avg_score)
    
    return np.mean(scores) if scores else 0.0


def calculate_all_metrics(references, predictions):
    """Calculate all text similarity metrics."""
    metrics = {}
    
    # Simple text comparison metrics
    metrics['exact_match'] = calculate_exact_match(references, predictions)
    metrics['levenshtein_similarity'] = calculate_levenshtein_similarity(references, predictions)
    metrics['jaccard_similarity'] = calculate_jaccard_similarity(references, predictions)
    metrics['tfidf_cosine_similarity'] = calculate_tfidf_cosine_similarity(references, predictions)
    
    # ROUGE scores
    rouge_scores = calculate_rouge_scores(references, predictions)
    metrics['rouge1'] = rouge_scores['rouge1']
    metrics['rouge2'] = rouge_scores['rouge2']
    metrics['rougeL'] = rouge_scores['rougeL']
    
    # BLEU scores
    bleu_scores = calculate_bleu_score(references, predictions)
    metrics['bleu1'] = bleu_scores['bleu1']
    metrics['bleu4'] = bleu_scores['bleu4']
    
    # METEOR score
    metrics['meteor'] = calculate_meteor_score(references, predictions)
    
    # Semantic similarity metrics
    metrics['spacy_similarity'] = calculate_spacy_similarity(references, predictions)
    
    # Sentence-BERT similarity
    if BERT_AVAILABLE:
        metrics['sbert_similarity'] = calculate_sbert_similarity(references, predictions)
    else:
        metrics['sbert_similarity'] = 0.0
    
    # Word Mover's Distance
    if WMD_AVAILABLE:
        metrics['wmd_similarity'] = calculate_word_movers_distance(references, predictions)
    else:
        metrics['wmd_similarity'] = 0.0
    
    # Entailment score
    if NLI_AVAILABLE:
        metrics['entailment_score'] = calculate_entailment_score(references, predictions)
    else:
        metrics['entailment_score'] = 0.0
    
    return metrics


def contains_code(text):
    """Check if the text contains code blocks."""
    return bool(re.search(r'```|\'\'\'', text))


def calculate_metrics_for_groups(references, predictions):
    """Calculate metrics separately for responses with code and without code."""
    # Separate responses with code and without code
    code_refs = []
    code_preds = []
    non_code_refs = []
    non_code_preds = []
    
    for ref, pred in zip(references, predictions):
        # Only check reference for code, not prediction
        if contains_code(ref):
            code_refs.append(ref)
            code_preds.append(pred)
        else:
            non_code_refs.append(ref)
            non_code_preds.append(pred)
    
    results = {
        "all": {
            "count": len(references),
            "metrics": calculate_all_metrics(references, predictions) if references else {}
        },
        "code": {
            "count": len(code_refs),
            "metrics": calculate_all_metrics(code_refs, code_preds) if code_refs else {}
        },
        "non_code": {
            "count": len(non_code_refs),
            "metrics": calculate_all_metrics(non_code_refs, non_code_preds) if non_code_refs else {}
        }
    }
    
    return results


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(i) for i in obj)
    else:
        return obj


def main():
    #output_dir = os.path.join(os.path.dirname(__file__), "outputsllama3.370b")
    directories=['outputs_llama3.3','outputs_qwen','outputs_phi','outputs_gemma']
    
    output_dir = os.path.join(os.path.dirname(__file__), "outputs_llama3.3")
    #output_dir = os.path.join(os.path.dirname(__file__), "outputs_qwen")


    # Read all csv files results_epoch_*.csv
    # Transform that to json and save it raw_responses_epoch_*.jsonl
    csv_pattern = os.path.join(output_dir, "results_epoch_*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if csv_files:
        print(f"Found {len(csv_files)} CSV result files to convert to JSONL")
        
        for csv_file in csv_files:
            # Extract epoch number from filename
            epoch_match = re.search(r"epoch_(\d+)", csv_file)
            if not epoch_match:
                print(f"Could not extract epoch number from {csv_file}, skipping")
                continue
                
            epoch = epoch_match.group(1)
            jsonl_file = os.path.join(output_dir, f"raw_responses_epoch_{epoch}.jsonl")
            
            # Skip if JSONL file already exists
            if os.path.exists(jsonl_file):
                print(f"JSONL file already exists for epoch {epoch}, skipping conversion")
                continue
            
            print(f"Converting {csv_file} to {jsonl_file}")
            
            try:
                # Read CSV file
                df = pd.read_csv(csv_file)
                
                # Check for required columns
                required_cols = ['sample_id', 'actual', 'predicted']
                if not all(col in df.columns for col in required_cols):
                    print(f"Required columns missing in {csv_file}, skipping")
                    continue
                #exit(0)
                # Write to JSONL file
                with open(jsonl_file, 'w', encoding='utf-8') as f:
                    for _, row in df.iterrows():
                        # Skip rows where either actual or predicted is NaN
                        if row['actual']=='nan' or row['predicted']=='nan':
                            print(f"Skipping row with NaN values in {csv_file}")
                            continue
                            
                        # Create JSON entry with reference instead of actual
                        json_entry = {
                            "sample_id": row['sample_id'],
                            "reference": row['actual'],
                            "prediction": row['predicted']
                        }
                        f.write(json.dumps(json_entry) + '\n')
                    #exit(0)
                print(f"Successfully converted {csv_file} to {jsonl_file}")
            except Exception as e:
                print(f"Error converting {csv_file}: {str(e)}")
    else:
        print("No CSV files found to convert")

    # Find all response files for all epochs
    pattern = os.path.join(output_dir, "raw_responses_epoch_*.jsonl")
    response_files = glob.glob(pattern)
    
    if not response_files:
        print(f"No raw_responses_epoch_*.jsonl files found in {output_dir}")
        # Try to find files in the current directory if not found in outputs
        pattern = "raw_responses_epoch_*.jsonl"
        response_files = glob.glob(pattern)
        if not response_files:
            print("No raw_responses_epoch_*.jsonl files found in current directory either")
            return {}
    
    print(f"Found {len(response_files)} response files")
    
    # Process each file and calculate metrics
    results = {}
    for file_path in response_files:
        # Extract epoch number from filename
        epoch_match = re.search(r"epoch_(\d+)", file_path)
        if not epoch_match:
            print(f"Could not extract epoch number from {file_path}, skipping")
            continue
            
        epoch = int(epoch_match.group(1))
        print(f"Processing file for epoch {epoch}: {file_path}")
        
        responses = read_raw_responses(file_path)
        
        if not responses:
            print(f"No responses found in {file_path}")
            continue
            
        references = []
        predictions = []
        count = 0
        for resp in responses:
            resp["reference"] = str(resp["reference"])
            resp["prediction"] = str(resp["prediction"])
            
            count += 1
            if "reference" in resp and "prediction" in resp:
                print(count,len(resp["reference"]), len(resp["prediction"]))
                if  resp["reference"]=='nan' or resp["prediction"]=='nan':
                    print(resp["reference"], resp["prediction"])
                    print(f"Skipping response with NaN values in {file_path}")
                else:
                    references.append(resp["reference"])
                    predictions.append(resp["prediction"])
        print(f"Found {len(references)} valid reference/prediction pairs")
        print(f"Found {len(predictions)} valid reference/prediction pairs")

        if references and predictions :
            # Calculate metrics for all responses and separate by code/non-code
            metrics_results = calculate_metrics_for_groups(references, predictions)
            
            results[epoch] = {
                "file": file_path,
                "num_responses": len(responses),
                "all": metrics_results["all"],
                "code": metrics_results["code"],
                "non_code": metrics_results["non_code"]
            }
            
            # Print summary for this epoch
            print(f"Epoch {epoch}: ({len(responses)} total responses)")
            print(f"  All: Exact Match = {metrics_results['all']['metrics'].get('exact_match', 0):.4f}, "
                  f"BLEU-1 = {metrics_results['all']['metrics'].get('bleu1', 0):.4f}, "
                  f"METEOR = {metrics_results['all']['metrics'].get('meteor', 0):.4f}")
            
            if metrics_results["code"]["count"] > 0:
                print(f"  Code ({metrics_results['code']['count']} responses): "
                      f"Exact Match = {metrics_results['code']['metrics'].get('exact_match', 0):.4f}, "
                      f"BLEU-1 = {metrics_results['code']['metrics'].get('bleu1', 0):.4f}")
            
            if metrics_results["non_code"]["count"] > 0:
                print(f"  Non-Code ({metrics_results['non_code']['count']} responses): "
                      f"Exact Match = {metrics_results['non_code']['metrics'].get('exact_match', 0):.4f}, "
                      f"BLEU-1 = {metrics_results['non_code']['metrics'].get('bleu1', 0):.4f}")
        else:
            print(f"No valid reference/prediction pairs found in {file_path}")
    
    # Create and display formatted tables
    if results:
        # List of metrics to display in table
        metric_names = [
            ('exact_match', 'Exact Match'),
            ('levenshtein_similarity', 'Levenshtein Sim'),
            ('jaccard_similarity', 'Jaccard Sim'),
            ('tfidf_cosine_similarity', 'TF-IDF Cosine'),
            ('rouge1', 'ROUGE-1'),
            ('rouge2', 'ROUGE-2'),
            ('rougeL', 'ROUGE-L'),
            ('bleu1', 'BLEU-1'),
            ('bleu4', 'BLEU-4'),
            ('meteor', 'METEOR'),
            ('spacy_similarity', 'spaCy Sim'),
            ('sbert_similarity', 'SBERT Sim'),
            ('wmd_similarity', 'WMD Sim'),
            ('entailment_score', 'Entailment')
        ]
        
        # Create tables for all, code, and non-code responses
        for category in ['all', 'code', 'non_code']:
            # Create table
            table = PrettyTable()
            table.field_names = ['Epoch'] + [display_name for _, display_name in metric_names]
            
            # Create DataFrame for saving as CSV
            df_data = []
            
            # Add rows
            for epoch in sorted(results.keys()):
                r = results[epoch]
                if category in r and r[category]["count"] > 0:
                    metrics = r[category]["metrics"]
                    row = [epoch]
                    row_dict = {'Epoch': epoch}
                    
                    for metric_key, display_name in metric_names:
                        value = metrics.get(metric_key, 0)
                        row.append(f"{value:.4f}")
                        row_dict[display_name] = value
                    
                    table.add_row(row)
                    df_data.append(row_dict)
            
            # Print table
            print(f"\n{category.title()} Responses - Detailed Evaluation Metrics by Epoch:")
            print(table)
            
            # Save table to file
            os.makedirs(output_dir, exist_ok=True)
            table_file = os.path.join(output_dir, f"detailed_metrics_{category}.txt")
            with open(table_file, "w") as f:
                f.write(str(table))
            
            # Save as pandas DataFrame and CSV
            if df_data:
                df = pd.DataFrame(df_data)
                
                # Save DataFrame to pickle file
                pandas_file = os.path.join(output_dir, f"detailed_metrics_{category}.pkl")
                df.to_pickle(pandas_file)
                
                # Save DataFrame to CSV
                csv_file = os.path.join(output_dir, f"detailed_metrics_{category}.csv")
                df.to_csv(csv_file, index=False)
                
                print(f"Metrics for {category} saved as DataFrame and CSV")
        
        # Save detailed results as JSON
        detailed_results_file = os.path.join(output_dir, "detailed_metrics_results.json")
        with open(detailed_results_file, "w") as f:
            # Convert numpy types to native Python types before serializing
            serializable_results = convert_numpy_types(results)
            json.dump(serializable_results, f, indent=2)
        
        print(f"Tables and detailed results saved to {output_dir}")
    else:
        print("No results were calculated")
    
    return results

if __name__ == "__main__":
    results = main()