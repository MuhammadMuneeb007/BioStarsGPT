import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import re
import ast
from typing import List, Dict, Any, Optional, Tuple, Union
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import nltk
from nltk.tokenize import word_tokenize
from nltk.metrics import edit_distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import os
from collections import defaultdict
import google.generativeai as genai
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TextStreamer, 
    RobertaTokenizer, RobertaModel, AutoModel
)

# Set up NLTK
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize availability flags
SPACY_AVAILABLE = False
BERT_AVAILABLE = False
WMD_AVAILABLE = False
NLI_AVAILABLE = False
BERTSCORE_AVAILABLE = False
CODEBERT_AVAILABLE = False

# Setup paths
os.environ['GENSIM_DATA_DIR'] = '/data/ascher02/uqmmune1/BioStarsGPT/temp/'
os.environ['TRANSFORMERS_CACHE'] = '/data/ascher02/uqmmune1/BioStarsGPT/temp/huggingface'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/data/ascher02/uqmmune1/BioStarsGPT/temp/sentence_transformers'

# Import and initialize optional components
try:
    from sentence_transformers import SentenceTransformer
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    BERT_AVAILABLE = True
except ImportError:
    warnings.warn("Sentence-BERT not available. SBERT metrics will be disabled.")

try:
    import gensim.downloader as api
    word2vec_model = api.load("word2vec-google-news-300")
    WMD_AVAILABLE = True
except ImportError:
    warnings.warn("Word2Vec not available. WMD metrics will be disabled.")
    word2vec_model = None

try:
    import spacy
    spacy_nlp = spacy.load('en_core_web_md')
    SPACY_AVAILABLE = True
except ImportError:
    warnings.warn("spaCy not available. Semantic similarity metrics will be disabled.")

try:
    from transformers import pipeline
    nli_pipeline = pipeline("zero-shot-classification", 
                           model="facebook/bart-large-mnli", 
                           device=0 if torch.cuda.is_available() else -1)
    NLI_AVAILABLE = True
except ImportError:
    warnings.warn("NLI model not available. Entailment scores will be disabled.")

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    warnings.warn("BERTScore not available. BERTScore metrics will be disabled.")

try:
    # Load CodeBERT model and tokenizer
    codebert_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    codebert_model = AutoModel.from_pretrained("microsoft/codebert-base")
    CODEBERT_AVAILABLE = True
except ImportError:
    warnings.warn("CodeBERT not available. CodeBERT metrics will be disabled.")

# Model types
class ModelType:
    GOOGLE_GENAI = "google_genai"
    HUGGINGFACE = "huggingface"

def load_jsonl_data(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                if isinstance(item, dict) and "prompt" in item and "completion" in item:
                    data.append(item)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line: {line}")
    
    print(f"Loaded {len(data)} valid items from {file_path}")
    return data

def load_sample_data() -> List[Dict[str, Any]]:
    """Load sample data from strings for testing."""
    sample_data = [
        {"prompt": "What is the scale of the latest dbSNP build?", 
         "completion": "The user is asking about the size and scope of the most recent dbSNP database. To understand this, they are looking for the highest or total number of 'rs' accession numbers, which are unique identifiers for single nucleotide polymorphisms (SNPs) in the dbSNP database. This number serves as an indicator of the database's scale and the extent of genetic variations cataloged within it. By knowing this number, researchers can grasp the comprehensiveness of the latest dbSNP build.",
         "complexity": "Unknown"},
        {"prompt": "How to determine the current scale of dbSNP by finding the total RS IDs in the latest build?", 
         "completion": "You can determine the current scale of the dbSNP database by looking at the total count of 'rs' accession numbers. The provided information indicates that the total number of RS IDs in the latest dbSNP build is approximately 1,130,597,309. This number represents the cumulative count of unique SNP identifiers cataloged in dbSNP up to the latest build. To get an idea of the scale, you can consider this number as the total number of variations currently indexed. Therefore, the scale of the latest dbSNP build, in terms of RS IDs, is over one billion. This information helps researchers understand the vastness of the database and the amount of genetic variation data it encompasses.",
         "complexity": "Unknown"}
    ]
    return sample_data

def load_huggingface_model(model_id: str, load_in_8bit: bool = False, device_map: str = "auto") -> Tuple[Any, Any]:
    """Load a model and tokenizer from Hugging Face."""
    print(f"Loading HuggingFace model: {model_id}")
    
    # Determine if we can use BF16 or FP16
    bf16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Load model with appropriate quantization
    if load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            load_in_8bit=True,
            torch_dtype=torch.float16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch.bfloat16 if bf16_supported else torch.float16
        )
    
    return model, tokenizer

def format_prompt(prompt: str, model_id: str) -> str:
    """Format the prompt according to the model's expected format."""
    # For Llama models
    if "llama" in model_id.lower():
        return f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|start_header_id|>assistant<|end_header_id|>\n\n"
    # For Mistral models
    elif "mistral" in model_id.lower():
        return f"<s>[INST] {prompt} [/INST]"
    # For Claude-like models
    elif "claude" in model_id.lower():
        return f"Human: {prompt}\n\nAssistant:"
    # For Phi models
    elif "phi" in model_id.lower():
        return f"User: {prompt}\n\nAssistant:"
    # For Qwen models
    elif "qwen" in model_id.lower():
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    # Default format for other models
    else:
        return f"USER: {prompt}\nASSISTANT:"

def generate_completion_with_huggingface(model, tokenizer, prompt: str, model_id: str, 
                                        max_new_tokens: int = 1024, temperature: float = 0.7, 
                                        top_p: float = 0.95) -> str:
    """Generate a completion for a prompt using a loaded HuggingFace model."""
    formatted_prompt = format_prompt(prompt, model_id)
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    streamer = TextStreamer(tokenizer)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            streamer=streamer
        )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the assistant's response based on model type
    if "llama" in model_id.lower():
        completion = full_output.split("assistant\n\n")[-1]
    elif "mistral" in model_id.lower():
        completion = full_output.split("[/INST]")[-1].strip()
    elif "claude" in model_id.lower():
        completion = full_output.split("Assistant:")[-1].strip()
    elif "phi" in model_id.lower():
        completion = full_output.split("Assistant:")[-1].strip()
    elif "qwen" in model_id.lower():
        completion = full_output.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
    else:
        completion = full_output.split("ASSISTANT:")[-1].strip()
    
    return completion

def generate_completion_with_genai(prompt: str, model_name: str, api_key: str, 
                                  temperature: float = 0.7, max_tokens: int = 1024) -> str:
    """Generate a completion for a prompt using Google Generative AI."""
    # Configure genai with the API key
    genai.configure(api_key=api_key)
    
    # Get the model
    model = genai.GenerativeModel(model_name)
    
    # Define generation config
    generation_config = {
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": max_tokens,
    }
    
    # Add instruction to prompt
    enhanced_prompt = f"""Please provide only the direct answer to this question without any additional text or explanations:

{prompt}"""
    
    # Generate the completion
    response = model.generate_content(enhanced_prompt, generation_config=generation_config)
    return response.text

def generate_completion(
    model_type: str,
    prompt: str,
    model_name: str,
    model: Any = None,
    tokenizer: Any = None,
    api_key: str = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.95
) -> str:
    """Generate completion using either HuggingFace or Google Generative AI models."""
    if model_type == ModelType.GOOGLE_GENAI:
        return generate_completion_with_genai(
            prompt=prompt,
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )
    elif model_type == ModelType.HUGGINGFACE:
        return generate_completion_with_huggingface(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            model_id=model_name,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def extract_code_blocks(text: str) -> List[str]:
    """Extract code blocks from text using markdown format."""
    # Match code blocks with or without language specification
    # e.g., ```python ... ``` or ``` ... ```
    pattern = r'```(?:\w+)?\s*(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

def is_valid_python_code(code_str: str) -> bool:
    """Check if a string is valid Python code."""
    try:
        ast.parse(code_str)
        return True
    except SyntaxError:
        return False

def calculate_codebert_similarity(references: List[str], predictions: List[str]) -> float:
    """Calculate CodeBERT-based semantic similarity between text containing code."""
    if not CODEBERT_AVAILABLE:
        return 0.0
    
    similarities = []
    
    for ref, pred in zip(references, predictions):
        if not ref.strip() or not pred.strip():
            continue
        
        # Tokenize texts
        ref_tokens = codebert_tokenizer(ref, return_tensors='pt', padding=True, truncation=True, max_length=512)
        pred_tokens = codebert_tokenizer(pred, return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        # Generate embeddings
        with torch.no_grad():
            ref_outputs = codebert_model(**ref_tokens)
            pred_outputs = codebert_model(**pred_tokens)
        
        # Get the [CLS] token embedding which represents the entire sequence
        ref_embedding = ref_outputs.last_hidden_state[:, 0, :]
        pred_embedding = pred_outputs.last_hidden_state[:, 0, :]
        
        # Calculate cosine similarity
        cosine_sim = torch.nn.functional.cosine_similarity(ref_embedding, pred_embedding)
        similarities.append(cosine_sim.item())
    
    return sum(similarities) / len(similarities) if similarities else 0

def calculate_bleu_codenn(references: List[str], predictions: List[str]) -> float:
    """
    Calculate BLEU-codeNN score which is an adaptation of BLEU for code with natural language.
    This gives higher weight to code tokens.
    """
    smooth = SmoothingFunction().method1
    scores = []
    
    for ref, pred in zip(references, predictions):
        if not ref.strip() or not pred.strip():
            continue
            
        # Extract code blocks
        ref_code_blocks = extract_code_blocks(ref)
        pred_code_blocks = extract_code_blocks(pred)
        
        # If no code blocks found, fall back to traditional BLEU
        if not ref_code_blocks and not pred_code_blocks:
            ref_tokens = word_tokenize(ref.lower())
            pred_tokens = word_tokenize(pred.lower())
            bleu = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
            scores.append(bleu)
            continue
        
        # Combine all code blocks
        ref_code = " ".join(ref_code_blocks)
        pred_code = " ".join(pred_code_blocks)
        
        # Tokenize code
        ref_code_tokens = word_tokenize(ref_code.lower())
        pred_code_tokens = word_tokenize(pred_code.lower())
        
        # Replace code blocks with placeholders in the original text
        ref_text = re.sub(r'```(?:\w+)?\s*.*?```', '[CODE]', ref, flags=re.DOTALL)
        pred_text = re.sub(r'```(?:\w+)?\s*.*?```', '[CODE]', pred, flags=re.DOTALL)
        
        # Tokenize the text without code blocks
        ref_text_tokens = word_tokenize(ref_text.lower())
        pred_text_tokens = word_tokenize(pred_text.lower())
        
        # Calculate BLEU separately for code and text
        if ref_code_tokens and pred_code_tokens:
            code_bleu = sentence_bleu([ref_code_tokens], pred_code_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
        else:
            code_bleu = 0
            
        if ref_text_tokens and pred_text_tokens:
            text_bleu = sentence_bleu([ref_text_tokens], pred_text_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
        else:
            text_bleu = 0
        
        # Weight code more heavily (0.7 for code, 0.3 for text)
        combined_bleu = (0.7 * code_bleu) + (0.3 * text_bleu)
        scores.append(combined_bleu)
    
    return sum(scores) / len(scores) if scores else 0

def calculate_code_keyword_match(references: List[str], predictions: List[str]) -> float:
    """Calculate match rate of important code keywords."""
    # Common keywords in programming languages
    keywords = set([
        "if", "else", "for", "while", "return", "function", "def", "class", "import",
        "from", "try", "except", "finally", "with", "as", "in", "not", "and", "or",
        "break", "continue", "pass", "lambda", "async", "await", "yield"
    ])
    
    scores = []
    for ref, pred in zip(references, predictions):
        # Extract code blocks
        ref_code_blocks = extract_code_blocks(ref)
        pred_code_blocks = extract_code_blocks(pred)
        
        if not ref_code_blocks:
            # No code in reference
            continue
            
        # Get keywords from reference code
        ref_keywords = set()
        for block in ref_code_blocks:
            for line in block.split('\n'):
                tokens = word_tokenize(line.lower())
                ref_keywords.update(keyword for keyword in tokens if keyword in keywords)
        
        # Get keywords from prediction code
        pred_keywords = set()
        for block in pred_code_blocks:
            for line in block.split('\n'):
                tokens = word_tokenize(line.lower())
                pred_keywords.update(keyword for keyword in tokens if keyword in keywords)
        
        if not ref_keywords:
            # No keywords found in reference code
            continue
            
        # Calculate Jaccard similarity of keywords
        intersection = ref_keywords.intersection(pred_keywords)
        union = ref_keywords.union(pred_keywords)
        score = len(intersection) / len(union) if union else 0
        scores.append(score)
    
    return sum(scores) / len(scores) if scores else 0

def calculate_hybrid_score(references: List[str], predictions: List[str]) -> float:
    """
    Calculate a hybrid score that combines text similarity with code similarity.
    This gives equal weight to code and natural language components.
    """
    text_similarities = []
    code_similarities = []
    
    for ref, pred in zip(references, predictions):
        if not ref.strip() or not pred.strip():
            continue
        
        # Extract code blocks
        ref_code_blocks = extract_code_blocks(ref)
        pred_code_blocks = extract_code_blocks(pred)
        
        # Process natural language (text without code blocks)
        ref_text = re.sub(r'```(?:\w+)?\s*.*?```', '', ref, flags=re.DOTALL).strip()
        pred_text = re.sub(r'```(?:\w+)?\s*.*?```', '', pred, flags=re.DOTALL).strip()
        
        # Calculate text similarity using TF-IDF
        if ref_text and pred_text:
            vectorizer = TfidfVectorizer()
            try:
                tfidf_matrix = vectorizer.fit_transform([ref_text, pred_text])
                text_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                text_similarities.append(text_sim)
            except ValueError:
                # Handle empty result from vectorizer
                pass
        
        # Calculate code similarity
        if ref_code_blocks and pred_code_blocks:
            # Join all code blocks
            ref_code = '\n'.join(ref_code_blocks)
            pred_code = '\n'.join(pred_code_blocks)
            
            # Simple token overlap for code
            ref_tokens = set(word_tokenize(ref_code.lower()))
            pred_tokens = set(word_tokenize(pred_code.lower()))
            
            if ref_tokens and pred_tokens:
                intersection = ref_tokens.intersection(pred_tokens)
                union = ref_tokens.union(pred_tokens)
                code_sim = len(intersection) / len(union)
                code_similarities.append(code_sim)
    
    # Calculate average similarities
    avg_text_sim = sum(text_similarities) / len(text_similarities) if text_similarities else 0
    avg_code_sim = sum(code_similarities) / len(code_similarities) if code_similarities else 0
    
    # Equal weighting for text and code
    return (avg_text_sim + avg_code_sim) / 2

def calculate_exact_match(references: List[str], predictions: List[str]) -> float:
    """Calculate exact match score."""
    exact_matches = sum(1 for ref, pred in zip(references, predictions) if ref.strip() == pred.strip())
    return exact_matches / len(references) if len(references) > 0 else 0

def calculate_levenshtein_similarity(references: List[str], predictions: List[str]) -> float:
    """Calculate normalized Levenshtein similarity."""
    similarities = []
    for ref, pred in zip(references, predictions):
        if len(ref) == 0 and len(pred) == 0:
            similarities.append(1.0)
        else:
            distance = edit_distance(ref, pred)
            max_len = max(len(ref), len(pred))
            similarities.append(1 - (distance / max_len))
    return sum(similarities) / len(similarities) if similarities else 0

def calculate_jaccard_similarity(references: List[str], predictions: List[str]) -> float:
    """Calculate Jaccard similarity."""
    similarities = []
    for ref, pred in zip(references, predictions):
        ref_tokens = set(word_tokenize(ref.lower()))
        pred_tokens = set(word_tokenize(pred.lower()))
        
        if not ref_tokens and not pred_tokens:
            similarities.append(1.0)
        else:
            intersection = ref_tokens.intersection(pred_tokens)
            union = ref_tokens.union(pred_tokens)
            similarities.append(len(intersection) / len(union) if union else 0)
    
    return sum(similarities) / len(similarities) if similarities else 0

def calculate_tfidf_cosine_similarity(references: List[str], predictions: List[str]) -> float:
    """Calculate TF-IDF cosine similarity."""
    if not references or not predictions:
        return 0.0
    
    # Combine references and predictions for the vectorizer
    all_texts = references + predictions
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(all_texts)
    except ValueError:
        return 0.0
    
    # Split the matrix back into references and predictions
    ref_vectors = tfidf_matrix[:len(references)]
    pred_vectors = tfidf_matrix[len(references):]
    
    # Calculate cosine similarity for each reference-prediction pair
    similarities = []
    for i in range(len(references)):
        if i < len(predictions):  # Ensure we have both reference and prediction
            similarity = cosine_similarity(ref_vectors[i], pred_vectors[i])[0][0]
            similarities.append(similarity)
    
    return sum(similarities) / len(similarities) if similarities else 0

def calculate_rouge_scores(references: List[str], predictions: List[str]) -> Dict[str, float]:
    """Calculate ROUGE scores."""
    rouge = Rouge()
    scores = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    valid_pairs = 0
    rouge1_sum = rouge2_sum = rougeL_sum = 0.0
    
    for ref, pred in zip(references, predictions):
        if not ref.strip() or not pred.strip():
            continue
        
        try:
            rouge_scores = rouge.get_scores(pred, ref)[0]
            rouge1_sum += rouge_scores['rouge-1']['f']
            rouge2_sum += rouge_scores['rouge-2']['f']
            rougeL_sum += rouge_scores['rouge-l']['f']
            valid_pairs += 1
        except Exception as e:
            print(f"Error calculating ROUGE scores: {e}")
    
    if valid_pairs > 0:
        scores["rouge1"] = rouge1_sum / valid_pairs
        scores["rouge2"] = rouge2_sum / valid_pairs
        scores["rougeL"] = rougeL_sum / valid_pairs
    
    return scores

def calculate_bleu_score(references: List[str], predictions: List[str]) -> Dict[str, float]:
    """Calculate BLEU scores."""
    smooth = SmoothingFunction().method1
    bleu1_sum = bleu4_sum = 0.0
    valid_pairs = 0
    
    for ref, pred in zip(references, predictions):
        if not ref.strip() or not pred.strip():
            continue
        
        ref_tokens = word_tokenize(ref.lower())
        pred_tokens = word_tokenize(pred.lower())
        
        if not pred_tokens:
            continue
        
        bleu1 = sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth)
        bleu4 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
        
        bleu1_sum += bleu1
        bleu4_sum += bleu4
        valid_pairs += 1
    
    return {
        "bleu1": bleu1_sum / valid_pairs if valid_pairs > 0 else 0,
        "bleu4": bleu4_sum / valid_pairs if valid_pairs > 0 else 0
    }

def calculate_meteor_score(references: List[str], predictions: List[str]) -> float:
    """Calculate METEOR score using NLTK."""
    from nltk.translate.meteor_score import meteor_score
    
    scores = []
    for ref, pred in zip(references, predictions):
        if not ref.strip() or not pred.strip():
            continue
            
        ref_tokens = word_tokenize(ref.lower())
        pred_tokens = word_tokenize(pred.lower())
        
        if not pred_tokens:
            continue
            
        score = meteor_score([ref_tokens], pred_tokens)
        scores.append(score)
    
    return sum(scores) / len(scores) if scores else 0

def calculate_spacy_similarity(references: List[str], predictions: List[str]) -> float:
    """Calculate spaCy similarity."""
    if not SPACY_AVAILABLE:
        return 0.0
    
    similarities = []
    for ref, pred in zip(references, predictions):
        if not ref.strip() or not pred.strip():
            continue
            
        ref_doc = spacy_nlp(ref)
        pred_doc = spacy_nlp(pred)
        
        if not ref_doc.vector_norm or not pred_doc.vector_norm:
            continue
            
        similarity = ref_doc.similarity(pred_doc)
        similarities.append(similarity)
    
    return sum(similarities) / len(similarities) if similarities else 0

def calculate_sbert_similarity(references: List[str], predictions: List[str]) -> float:
    """Calculate Sentence-BERT cosine similarity."""
    if not BERT_AVAILABLE:
        return 0.0
    
    similarities = []
    for ref, pred in zip(references, predictions):
        if not ref.strip() or not pred.strip():
            continue
            
        ref_embedding = sbert_model.encode([ref], convert_to_tensor=True)
        pred_embedding = sbert_model.encode([pred], convert_to_tensor=True)
        
        # Calculate cosine similarity
        cosine_scores = torch.nn.functional.cosine_similarity(ref_embedding, pred_embedding)
        similarities.append(cosine_scores.item())
    
    return sum(similarities) / len(similarities) if similarities else 0

def calculate_word_movers_distance(references: List[str], predictions: List[str]) -> float:
    """Calculate normalized Word Mover's Distance similarity."""
    if not WMD_AVAILABLE or word2vec_model is None:
        return 0.0
    
    similarities = []
    for ref, pred in zip(references, predictions):
        try:
            ref_tokens = [w for w in word_tokenize(ref.lower()) if w in word2vec_model.key_to_index]
            pred_tokens = [w for w in word_tokenize(pred.lower()) if w in word2vec_model.key_to_index]
            
            if not ref_tokens or not pred_tokens:
                continue
            
            # Calculate WMD distance (lower is better)
            distance = word2vec_model.wmdistance(ref_tokens, pred_tokens)
            # Convert to similarity (higher is better)
            similarity = 1 / (1 + distance) if distance != float('inf') else 0
            similarities.append(similarity)
        except Exception as e:
            print(f"WMD error: {e}")
    
    return sum(similarities) / len(similarities) if similarities else 0

def calculate_entailment_score(references: List[str], predictions: List[str]) -> float:
    """Calculate NLI-based entailment score."""
    if not NLI_AVAILABLE or nli_pipeline is None:
        return 0.0
    
    entailment_scores = []
    batch_size = 8  # Process in batches to avoid memory issues
    
    for i in range(0, len(references), batch_size):
        batch_refs = references[i:i+batch_size]
        batch_preds = predictions[i:i+batch_size]
        
        for ref, pred in zip(batch_refs, batch_preds):
            if not ref.strip() or not pred.strip():
                continue
                
            # Check if prediction entails reference
            try:
                result = nli_pipeline(pred, hypothesis_template="{}", candidate_labels=["contradiction", "neutral", "entailment"])
                entailment_idx = result['labels'].index('entailment')
                entailment_score = result['scores'][entailment_idx]
                entailment_scores.append(entailment_score)
            except Exception as e:
                print(f"Entailment error: {e}")
    
    return sum(entailment_scores) / len(entailment_scores) if entailment_scores else 0
def calculate_bert_score(references: List[str], predictions: List[str]) -> Dict[str, float]:
    """Calculate BERTScore metrics."""
    if not BERTSCORE_AVAILABLE:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    # Filter out empty strings
    valid_pairs = [(ref, pred) for ref, pred in zip(references, predictions) 
                   if ref.strip() and pred.strip()]
    
    if not valid_pairs:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    valid_refs, valid_preds = zip(*valid_pairs)
    
    # Calculate BERTScore
    P, R, F1 = bert_score(valid_preds, valid_refs, lang='en', verbose=False)
    
    # Convert to float values
    precision = torch.mean(P).item()
    recall = torch.mean(R).item()
    f1 = torch.mean(F1).item()
    
    return {"precision": precision, "recall": recall, "f1": f1}

def calculate_metrics(references: List[str], predictions: List[str]) -> Dict[str, float]:
    """Calculate all text similarity metrics, including code-specific metrics."""
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
    
    # Code-specific metrics
    metrics['bleu_codenn'] = calculate_bleu_codenn(references, predictions)
    metrics['code_keyword_match'] = calculate_code_keyword_match(references, predictions)
    metrics['codebert_similarity'] = calculate_codebert_similarity(references, predictions)
    metrics['hybrid_text_code_score'] = calculate_hybrid_score(references, predictions)
    
    # METEOR score
    metrics['meteor'] = calculate_meteor_score(references, predictions)
    
    # Semantic similarity metrics
    metrics['spacy_similarity'] = calculate_spacy_similarity(references, predictions)
    metrics['sbert_similarity'] = calculate_sbert_similarity(references, predictions)
    metrics['wmd_similarity'] = calculate_word_movers_distance(references, predictions)
    
    # Entailment score
    metrics['entailment_score'] = calculate_entailment_score(references, predictions)
    
    # BERTScore
    bert_scores = calculate_bert_score(references, predictions)
    metrics['bertscore_precision'] = bert_scores['precision']
    metrics['bertscore_recall'] = bert_scores['recall'] 
    metrics['bertscore_f1'] = bert_scores['f1']
    
    return metrics

def plot_metrics(metrics: Dict[str, float], output_path: Optional[str] = None):
    """Plot metrics as a horizontal bar chart."""
    # Exclude certain metrics if not available
    filtered_metrics = {k: v for k, v in metrics.items() if k not in ['perplexity']}
    
    plt.figure(figsize=(12, 10))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Sort metrics by value
    sorted_metrics = sorted(filtered_metrics.items(), key=lambda x: x[1], reverse=True)
    labels, values = zip(*sorted_metrics)
    
    # Replace underscores with spaces and capitalize for better readability
    pretty_labels = [label.replace('_', ' ').title() for label in labels]
    
    # Create horizontal bar chart with categorized colors
    code_metrics = ['Bleu Codenn', 'Code Keyword Match', 'Codebert Similarity', 'Hybrid Text Code Score']
    colors = []
    for label in pretty_labels:
        if label in code_metrics:
            colors.append('#1f77b4')  # Blue for code metrics
        elif 'Bert' in label:
            colors.append('#ff7f0e')  # Orange for BERT-based metrics
        elif 'Bleu' in label or 'Rouge' in label:
            colors.append('#2ca02c')  # Green for BLEU/ROUGE metrics
        else:
            colors.append('#d62728')  # Red for other metrics
    
    bars = plt.barh(pretty_labels, values, color=colors)
    
    # Add value annotations
    for i, v in enumerate(values):
        plt.text(max(v + 0.02, 0.05), i, f'{v:.4f}', va='center')
    
    plt.xlabel('Score (higher is better)')
    plt.title('Model Evaluation Metrics')
    plt.xlim(0, 1.1)  # Most metrics are between 0 and 1
    plt.tight_layout()
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='Code Metrics'),
        Patch(facecolor='#ff7f0e', label='BERT-based Metrics'),
        Patch(facecolor='#2ca02c', label='BLEU/ROUGE Metrics'),
        Patch(facecolor='#d62728', label='Other Metrics')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    plt.show()

def save_results(data: List[Dict[str, Any]], predictions: List[str], metrics: Dict[str, float], 
                 output_dir: str, model_name: str):
    """Save results to files."""
    os.makedirs(output_dir, exist_ok=True)
    model_id_safe = model_name.replace('/', '_')
    
    # Save metrics to JSON
    metrics_path = os.path.join(output_dir, f"{model_id_safe}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save detailed results to CSV
    results_data = []
    for i, (item, pred) in enumerate(zip(data, predictions)):
        results_data.append({
            'id': i,
            'prompt': item['prompt'],
            'reference': item['completion'],
            'prediction': pred
        })
    
    results_df = pd.DataFrame(results_data)
    results_path = os.path.join(output_dir, f"{model_id_safe}_results.csv")
    results_df.to_csv(results_path, index=False)
    
    # Save metrics to CSV for easier comparison
    metrics_df = pd.DataFrame([metrics])
    metrics_csv_path = os.path.join(output_dir, f"{model_id_safe}_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    
    # Save plot
    plot_path = os.path.join(output_dir, f"{model_id_safe}_metrics_plot.png")
    plot_metrics(metrics, plot_path)
    
    print(f"Results saved to {output_dir}")
    return results_df, metrics_df

def main():
    # Configuration - Define available models of both types
    available_models = {
        
        # HuggingFace models
        "1": {"type": ModelType.HUGGINGFACE, "name": "unsloth/Llama-3.2-3B-Instruct"},
        "2": {"type": ModelType.HUGGINGFACE, "name": "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"},
        "3": {"type": ModelType.HUGGINGFACE, "name": "unsloth/Phi-4"},
        "4": {"type": ModelType.HUGGINGFACE, "name": "unsloth/Qwen2.5-7B"},
        "5": {"type": ModelType.HUGGINGFACE, "name": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"}
    }
    
    # Display available models
    print("Available models for evaluation:")
    for key, model_info in available_models.items():
        model_type = "Google Generative AI" if model_info["type"] == ModelType.GOOGLE_GENAI else "HuggingFace"
        print(f"{key}: {model_info['name']} ({model_type})")
    
    # Get user choice
    choice = "1"
    model_info = available_models.get(choice, available_models["5"])
    model_type = model_info["type"]
    model_name = model_info["name"]
    
    print(f"Using model: {model_name} ({model_type})")
    
    data_path = "Test_with_outliers.jsonl"  # Change to actual file path when available
    
    # Create model-specific output directory
    model_id = model_name.replace("/", "_")
    output_dir = os.path.join("Evaluations", model_id)
    max_samples = None  # Set to None to use all data
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    if os.path.exists(data_path):
        data = load_jsonl_data(data_path)
        if max_samples is not None:
            data = data[:max_samples]
    else:
        print(f"File {data_path} not found, using sample data")
        data = load_sample_data()
        # Save sample data to file for future use
        with open(os.path.join(output_dir, "sample_data.jsonl"), 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
    
    # Initialize model if needed
    model = tokenizer = None
    if model_type == ModelType.HUGGINGFACE:
        try:
            model, tokenizer = load_huggingface_model(model_name, load_in_8bit=True)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to smaller model")
            model_name = "mistralai/Mistral-7B-Instruct-v0.2"  # Fallback model
            model, tokenizer = load_huggingface_model(model_name, load_in_8bit=True)
    
    # Generate predictions
    prompts = [item["prompt"] for item in data]
    references = [item["completion"] for item in data]
    predictions = []
    
    print(f"Generating completions for {len(prompts)} prompts...")
    for i, prompt in enumerate(tqdm(prompts)):
        try:
            if model_type == ModelType.GOOGLE_GENAI:
                prediction = generate_completion(
                    model_type=model_type,
                    prompt=prompt,
                    model_name=model_name,
                    api_key=model_info.get("api_key")
                )
            else:  # HuggingFace model
                prediction = generate_completion(
                    model_type=model_type,
                    prompt=prompt,
                    model_name=model_name,
                    model=model, 
                    tokenizer=tokenizer
                )
            predictions.append(prediction)
            # Add delay to respect rate limits for API-based models
            if model_type == ModelType.GOOGLE_GENAI:
                time.sleep(1)
        except Exception as e:
            print(f"Error generating completion for prompt {i}: {e}")
            predictions.append("")
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(references, predictions)
    
    # Save results
    results_df, metrics_df = save_results(data, predictions, metrics, output_dir, model_id)
    
    # Print metrics summary
    print("\nMetrics Summary:")
    for metric_name, metric_value in sorted(metrics.items(), key=lambda x: x[1], reverse=True):
        print(f"{metric_name}: {metric_value:.4f}")
    
    # Return results for further analysis
    return {
        "model_type": model_type,
        "model_name": model_name,
        "metrics": metrics,
        "results_df": results_df,
        "metrics_df": metrics_df,
        "predictions": predictions,
        "references": references
    }

if __name__ == "__main__":
    results = main()