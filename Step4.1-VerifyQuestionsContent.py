import os
import json
import re
import difflib
import numpy as np
import pandas as pd
import csv
import sys  # Added for command-line arguments
from collections import Counter
from typing import List, Dict, Any, Tuple, Set, Optional, Union
import concurrent.futures  # Added for multithreading
import time  # Added to track processing time
from tqdm import tqdm  # Added for progress bars

# NLP libraries
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# For sentence embeddings
from sentence_transformers import SentenceTransformer

# For BERTScore
from bert_score import score as bert_score

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_md")
except:
    raise

# Load sentence transformer model
try:
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
except:
    raise

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# ================ TEXT PREPROCESSING FUNCTIONS ================

def preprocess_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

def tokenize_and_lemmatize(text: str) -> List[str]:
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    return [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]

# ================ SIMILARITY METRIC FUNCTIONS ================

def calculate_tfidf_similarity(text1: str, text2: str) -> float:
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity
    except Exception:
        return 0.0

def calculate_spacy_similarity(text1: str, text2: str) -> float:
    try:
        doc1 = nlp(text1)
        doc2 = nlp(text2)
        if not doc1.vector.any() or not doc2.vector.any():
            return 0.0
        return doc1.similarity(doc2)
    except Exception:
        return 0.0

def calculate_sentence_transformer_similarity(text1: str, text2: str) -> float:
    try:
        embeddings = sentence_model.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
        return similarity
    except Exception:
        return 0.0

def calculate_bertscore(text1: str, text2: str) -> Dict[str, float]:
    try:
        sentences1 = sent_tokenize(text1)
        sentences2 = sent_tokenize(text2)
        candidates = sentences1
        references = [sentences2] * len(candidates)
        P, R, F1 = bert_score(candidates, references, lang="en")
        return {
            "precision": P.mean().item(),
            "recall": R.mean().item(),
            "f1": F1.mean().item()
        }
    except Exception:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

def calculate_word_embedding_similarity(text1: str, text2: str) -> float:
    try:
        doc1 = nlp(text1)
        doc2 = nlp(text2)
        vec1 = np.mean([token.vector for token in doc1 if token.has_vector], axis=0) if len(doc1) > 0 else np.zeros(300)
        vec2 = np.mean([token.vector for token in doc2 if token.has_vector], axis=0) if len(doc2) > 0 else np.zeros(300)
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return similarity
    except Exception:
        return 0.0

def calculate_entity_similarity(text1: str, text2: str) -> Dict[str, float]:
    try:
        doc1 = nlp(text1)
        doc2 = nlp(text2)
        entities1 = {ent.text.lower(): ent.label_ for ent in doc1.ents}
        entities2 = {ent.text.lower(): ent.label_ for ent in doc2.ents}
        common_entities = set(entities1.keys()) & set(entities2.keys())
        label_matches = sum(1 for e in common_entities if entities1[e] == entities2[e])
        total_entities = len(entities1) + len(entities2)
        if total_entities == 0:
            return {"entity_overlap": 0.0, "label_match": 0.0}
        if len(common_entities) == 0:
            return {"entity_overlap": 0.0, "label_match": 0.0}
        return {
            "entity_overlap": len(common_entities) / (len(set(entities1.keys()) | set(entities2.keys()))),
            "label_match": label_matches / len(common_entities) if common_entities else 0.0
        }
    except Exception:
        return {"entity_overlap": 0.0, "label_match": 0.0}

def calculate_jaccard_similarity(text1: str, text2: str) -> float:
    try:
        tokens1 = set(tokenize_and_lemmatize(text1))
        tokens2 = set(tokenize_and_lemmatize(text2))
        if not tokens1 and not tokens2:
            return 0.0
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        return len(intersection) / len(union)
    except Exception:
        return 0.0

# ================ CONTENT ANALYSIS FUNCTIONS ================

def extract_key_concepts(text: str, top_n: int = 20) -> List[str]:
    try:
        doc = nlp(text)
        entities = [ent.text.lower() for ent in doc.ents]
        noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks]
        stop_words = set(stopwords.words('english'))
        important_words = [token.text.lower() for token in doc 
                          if token.is_alpha and token.text.lower() not in stop_words 
                          and token.pos_ in ('NOUN', 'PROPN', 'ADJ')]
        all_concepts = entities + noun_phrases + important_words
        concept_counter = Counter(all_concepts)
        return [concept for concept, count in concept_counter.most_common(top_n)]
    except Exception:
        return []

def identify_missing_content(explanation: str, processed_text: str, key_concepts: List[str]) -> List[str]:
    try:
        missing_concepts = []
        processed_doc = nlp(processed_text)
        for concept in key_concepts:
            concept_doc = nlp(concept)
            if concept.lower() in processed_text.lower():
                continue
            found_similar = False
            processed_sentences = [nlp(sent) for sent in sent_tokenize(processed_text)]
            for sent_doc in processed_sentences:
                if sent_doc.similarity(concept_doc) > 0.7:
                    found_similar = True
                    break
            if not found_similar:
                missing_concepts.append(concept)
        return missing_concepts
    except Exception:
        return []

def analyze_sentence_coverage(explanation: str, processed_text: str) -> Dict[str, Any]:
    try:
        explanation_sentences = sent_tokenize(explanation)
        processed_sentences = sent_tokenize(processed_text)
        coverage_metrics = {
            "sentence_coverage": [],
            "explanation_sentences_covered": set(),
            "potential_contradictions": []
        }
        for i, exp_sentence in enumerate(explanation_sentences):
            exp_doc = nlp(exp_sentence)
            max_similarity = 0
            most_similar_sentence_idx = -1
            for j, proc_sentence in enumerate(processed_sentences):
                proc_doc = nlp(proc_sentence)
                similarity = exp_doc.similarity(proc_doc)
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_sentence_idx = j
            if max_similarity > 0.5:
                coverage_metrics["sentence_coverage"].append({
                    "explanation_sentence": exp_sentence,
                    "processed_sentence": processed_sentences[most_similar_sentence_idx] if most_similar_sentence_idx >= 0 else "",
                    "similarity": max_similarity,
                })
                coverage_metrics["explanation_sentences_covered"].add(i)
            if 0 < max_similarity < 0.3 and most_similar_sentence_idx >= 0:
                exp_entities = set([ent.text.lower() for ent in exp_doc.ents])
                proc_entities = set([ent.text.lower() for ent in nlp(processed_sentences[most_similar_sentence_idx]).ents])
                common_entities = exp_entities.intersection(proc_entities)
                if len(common_entities) > 0:
                    coverage_metrics["potential_contradictions"].append({
                        "explanation_sentence": exp_sentence,
                        "processed_sentence": processed_sentences[most_similar_sentence_idx],
                        "common_entities": list(common_entities)
                    })
        if explanation_sentences:
            coverage_metrics["explanation_coverage_percent"] = (len(coverage_metrics["explanation_sentences_covered"]) / len(explanation_sentences)) * 100
        else:
            coverage_metrics["explanation_coverage_percent"] = 0.0
        return coverage_metrics
    except Exception:
        return {
            "sentence_coverage": [],
            "explanation_sentences_covered": set(),
            "potential_contradictions": [],
            "explanation_coverage_percent": 0.0
        }

# ================ MAIN COMPARISON FUNCTION ================

def compare_text_similarity(question_dir: str) -> Dict[str, Any]:
    results = {
        "directory": os.path.basename(question_dir),
        "tfidf_similarity_exp1": None,
        "tfidf_similarity_exp2": None,
        "spacy_similarity_exp1": None,
        "spacy_similarity_exp2": None,
        "sentence_transformer_similarity_exp1": None,
        "sentence_transformer_similarity_exp2": None,
        "word_embedding_similarity_exp1": None,
        "word_embedding_similarity_exp2": None,
        "bertscore_f1_exp1": None,
        "bertscore_precision_exp1": None, 
        "bertscore_recall_exp1": None,
        "bertscore_f1_exp2": None,
        "bertscore_precision_exp2": None,
        "bertscore_recall_exp2": None,
        "jaccard_similarity_exp1": None,
        "jaccard_similarity_exp2": None,
        "entity_overlap_exp1": None,
        "entity_overlap_exp2": None,
        "entity_label_match_exp1": None,
        "entity_label_match_exp2": None,
        "key_concepts_exp1": [],
        "key_concepts_exp2": [],
        "missing_concepts_exp1": [],
        "missing_concepts_exp2": [],
        "concept_coverage_percent_exp1": None,
        "concept_coverage_percent_exp2": None,
        "sentence_count_processed": 0,
        "sentence_count_exp1": 0,
        "sentence_count_exp2": 0,
        "explanation_coverage_percent_exp1": None,
        "explanation_coverage_percent_exp2": None,
        "potential_contradictions_exp1": [],
        "potential_contradictions_exp2": []
    }
    questions_json_path = os.path.join(question_dir, "Question.json")
    explanation1_text = ""
    explanation2_text = ""
    try:
        if os.path.exists(questions_json_path):
            with open(questions_json_path, 'r') as f:
                questions_data = json.load(f)
                if isinstance(questions_data, list):
                    for item in questions_data:
                        if "Explanation" in item:
                            explanation1_text = item["Explanation"]
                            break
                    for item in questions_data:
                        if "Explanation2" in item:
                            explanation2_text = item["Explanation2"]
                            break
                elif isinstance(questions_data, dict):
                    if "Explanation" in questions_data:
                        explanation1_text = questions_data["Explanation"]
                    if "Explanation2" in questions_data:
                        explanation2_text = questions_data["Explanation2"]
                else:
                    return results
        else:
            return results
    except Exception:
        return results
    processed_text_path = os.path.join(question_dir, "ProcessedText.md")
    processed_text = ""
    try:
        if os.path.exists(processed_text_path):
            with open(processed_text_path, 'r') as f:
                processed_text = f.read()
        else:
            return results
    except Exception:
        return results
    if not explanation1_text and not explanation2_text:
        return results
    results["sentence_count_processed"] = len(sent_tokenize(processed_text))
    if explanation1_text:
        results["sentence_count_exp1"] = len(sent_tokenize(explanation1_text))
        results["tfidf_similarity_exp1"] = calculate_tfidf_similarity(processed_text, explanation1_text)
        results["spacy_similarity_exp1"] = calculate_spacy_similarity(processed_text, explanation1_text)
        results["sentence_transformer_similarity_exp1"] = calculate_sentence_transformer_similarity(processed_text, explanation1_text)
        results["word_embedding_similarity_exp1"] = calculate_word_embedding_similarity(processed_text, explanation1_text)
        bertscore_results = calculate_bertscore(processed_text, explanation1_text)
        results["bertscore_f1_exp1"] = bertscore_results["f1"]
        results["bertscore_precision_exp1"] = bertscore_results["precision"]
        results["bertscore_recall_exp1"] = bertscore_results["recall"]
        results["jaccard_similarity_exp1"] = calculate_jaccard_similarity(processed_text, explanation1_text)
        entity_sim = calculate_entity_similarity(processed_text, explanation1_text)
        results["entity_overlap_exp1"] = entity_sim["entity_overlap"]
        results["entity_label_match_exp1"] = entity_sim["label_match"]
        results["key_concepts_exp1"] = extract_key_concepts(explanation1_text)
        results["missing_concepts_exp1"] = identify_missing_content(explanation1_text, processed_text, results["key_concepts_exp1"])
        if results["key_concepts_exp1"]:
            results["concept_coverage_percent_exp1"] = ((len(results["key_concepts_exp1"]) - len(results["missing_concepts_exp1"])) / len(results["key_concepts_exp1"])) * 100
        coverage_analysis = analyze_sentence_coverage(explanation1_text, processed_text)
        results["explanation_coverage_percent_exp1"] = coverage_analysis["explanation_coverage_percent"]
        results["potential_contradictions_exp1"] = [item["explanation_sentence"] for item in coverage_analysis["potential_contradictions"]]
    if explanation2_text:
        results["sentence_count_exp2"] = len(sent_tokenize(explanation2_text))
        results["tfidf_similarity_exp2"] = calculate_tfidf_similarity(processed_text, explanation2_text)
        results["spacy_similarity_exp2"] = calculate_spacy_similarity(processed_text, explanation2_text)
        results["sentence_transformer_similarity_exp2"] = calculate_sentence_transformer_similarity(processed_text, explanation2_text)
        results["word_embedding_similarity_exp2"] = calculate_word_embedding_similarity(processed_text, explanation2_text)
        bertscore_results = calculate_bertscore(processed_text, explanation2_text)
        results["bertscore_f1_exp2"] = bertscore_results["f1"]
        results["bertscore_precision_exp2"] = bertscore_results["precision"]
        results["bertscore_recall_exp2"] = bertscore_results["recall"]
        results["jaccard_similarity_exp2"] = calculate_jaccard_similarity(processed_text, explanation2_text)
        entity_sim = calculate_entity_similarity(processed_text, explanation2_text)
        results["entity_overlap_exp2"] = entity_sim["entity_overlap"]
        results["entity_label_match_exp2"] = entity_sim["label_match"]
        results["key_concepts_exp2"] = extract_key_concepts(explanation2_text)
        results["missing_concepts_exp2"] = identify_missing_content(explanation2_text, processed_text, results["key_concepts_exp2"])
        if results["key_concepts_exp2"]:
            results["concept_coverage_percent_exp2"] = ((len(results["key_concepts_exp2"]) - len(results["missing_concepts_exp2"])) / len(results["key_concepts_exp2"])) * 100
        coverage_analysis = analyze_sentence_coverage(explanation2_text, processed_text)
        results["explanation_coverage_percent_exp2"] = coverage_analysis["explanation_coverage_percent"]
        results["potential_contradictions_exp2"] = [item["explanation_sentence"] for item in coverage_analysis["potential_contradictions"]]
    return results

# ================ MULTIPLE QUESTIONS PROCESSING ================

def save_results_to_csv(results: List[Dict[str, Any]], output_path: str) -> None:
    if not results:
        print("No results to save")
        return
    columns = []
    for key, value in results[0].items():
        if isinstance(value, (str, int, float)) or value is None:
            columns.append(key)
        elif isinstance(value, list):
            columns.append(f"{key}_count")
            columns.append(f"{key}_sample")
    try:
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            for result in results:
                row = {}
                for key in columns:
                    if key in result:
                        row[key] = result[key]
                    elif key.endswith('_count') and key[:-6] in result:
                        row[key] = len(result[key[:-6]]) if result[key[:-6]] else 0
                    elif key.endswith('_sample') and key[:-7] in result:
                        original_key = key[:-7]
                        if result[original_key] and len(result[original_key]) > 0:
                            row[key] = ', '.join(str(x) for x in result[original_key][:3])
                            if len(result[original_key]) > 3:
                                row[key] += '...'
                        else:
                            row[key] = ''
                writer.writerow(row)
        print(f"\nResults saved to {output_path}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")

def save_results_to_json(result: Dict[str, Any], output_path: str) -> None:
    """
    Save a single result to a JSON file.
    
    Args:
        result: Dictionary containing the results
        output_path: Path to save the JSON file
    """
    try:
        # Convert any non-serializable types (like sets) to lists
        serializable_result = {}
        for key, value in result.items():
            if isinstance(value, set):
                serializable_result[key] = list(value)
            else:
                serializable_result[key] = value
                
        with open(output_path, 'w') as f:
            json.dump(serializable_result, f, indent=2)
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error saving results to JSON: {e}")

def calculate_statistics(results: List[Dict[str, Any]]) -> None:
    if not results:
        print("No results to analyze")
        return
    metrics = {
        "TF-IDF Similarity": {
            "Exp1": [r["tfidf_similarity_exp1"] for r in results if r.get("tfidf_similarity_exp1") is not None],
            "Exp2": [r["tfidf_similarity_exp2"] for r in results if r.get("tfidf_similarity_exp2") is not None]
        },
        "spaCy Similarity": {
            "Exp1": [r["spacy_similarity_exp1"] for r in results if r.get("spacy_similarity_exp1") is not None],
            "Exp2": [r["spacy_similarity_exp2"] for r in results if r.get("spacy_similarity_exp2") is not None]
        },
        "Sentence Transformer": {
            "Exp1": [r["sentence_transformer_similarity_exp1"] for r in results if r.get("sentence_transformer_similarity_exp1") is not None],
            "Exp2": [r["sentence_transformer_similarity_exp2"] for r in results if r.get("sentence_transformer_similarity_exp2") is not None]
        },
        "Word Embedding": {
            "Exp1": [r["word_embedding_similarity_exp1"] for r in results if r.get("word_embedding_similarity_exp1") is not None],
            "Exp2": [r["word_embedding_similarity_exp2"] for r in results if r.get("word_embedding_similarity_exp2") is not None]
        },
        "BERTScore F1": {
            "Exp1": [r["bertscore_f1_exp1"] for r in results if r.get("bertscore_f1_exp1") is not None],
            "Exp2": [r["bertscore_f1_exp2"] for r in results if r.get("bertscore_f1_exp2") is not None]
        },
        "Jaccard Similarity": {
            "Exp1": [r["jaccard_similarity_exp1"] for r in results if r.get("jaccard_similarity_exp1") is not None],
            "Exp2": [r["jaccard_similarity_exp2"] for r in results if r.get("jaccard_similarity_exp2") is not None]
        },
        "Entity Overlap": {
            "Exp1": [r["entity_overlap_exp1"] for r in results if r.get("entity_overlap_exp1") is not None],
            "Exp2": [r["entity_overlap_exp2"] for r in results if r.get("entity_overlap_exp2") is not None]
        },
        "Concept Coverage %": {
            "Exp1": [r["concept_coverage_percent_exp1"] for r in results if r.get("concept_coverage_percent_exp1") is not None],
            "Exp2": [r["concept_coverage_percent_exp2"] for r in results if r.get("concept_coverage_percent_exp2") is not None]
        },
        "Explanation Coverage %": {
            "Exp1": [r["explanation_coverage_percent_exp1"] for r in results if r.get("explanation_coverage_percent_exp1") is not None],
            "Exp2": [r["explanation_coverage_percent_exp2"] for r in results if r.get("explanation_coverage_percent_exp2") is not None]
        }
    }
    stats_data = []
    for metric_name, metric_data in metrics.items():
        for exp_name, values in metric_data.items():
            if values:
                stats_data.append({
                    'Metric': f"{metric_name} ({exp_name})",
                    'Min': min(values),
                    'Max': max(values), 
                    'Avg': sum(values)/len(values),
                    'Median': sorted(values)[len(values)//2],
                    'Count': len(values)
                })
    problem_counts = {
        "Missing Concepts": {
            "Exp1": sum(1 for r in results if r.get("missing_concepts_exp1") and len(r.get("missing_concepts_exp1")) > 0),
            "Exp2": sum(1 for r in results if r.get("missing_concepts_exp2") and len(r.get("missing_concepts_exp2")) > 0)
        },
        "Potential Contradictions": {
            "Exp1": sum(1 for r in results if r.get("potential_contradictions_exp1") and len(r.get("potential_contradictions_exp1")) > 0),
            "Exp2": sum(1 for r in results if r.get("potential_contradictions_exp2") and len(r.get("potential_contradictions_exp2")) > 0)
        }
    }
    for problem, exp_data in problem_counts.items():
        for exp_name, count in exp_data.items():
            stats_data.append({
                'Metric': f"Questions with {problem} ({exp_name})",
                'Min': None,
                'Max': None, 
                'Avg': None,
                'Median': None,
                'Count': count
            })
    stats_df = pd.DataFrame(stats_data)
    print("\n## Enhanced Similarity Statistics")
    print(stats_df.to_markdown(index=False, floatfmt=".4f"))

def process_question(subdir: str) -> Dict[str, Any]:
    return compare_text_similarity(subdir)

def process_multiple_questions(base_dir: str, output_csv: str, max_questions: Optional[int] = None, num_workers: int = None) -> None:
    start_time = time.time()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    questions_dir = os.path.join(script_dir, base_dir)
    if not os.path.exists(questions_dir):
        return
    try:
        subdirs = [os.path.join(questions_dir, d) for d in os.listdir(questions_dir) 
                  if os.path.isdir(os.path.join(questions_dir, d))]
    except Exception:
        return
    subdirs = sorted(subdirs)
    if max_questions:
        subdirs = subdirs[:max_questions]
    
    all_results = []
    print(f"Processing {len(subdirs)} question directories with {num_workers} workers...")
    
    # Create a progress bar
    with tqdm(total=len(subdirs), desc="Processing questions", unit="dir") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(process_question, subdir): subdir for subdir in subdirs}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                subdir = futures[future]
                try:
                    result = future.result()
                    if result:
                        all_results.append(result)
                except Exception as e:
                    print(f"Error processing {subdir}: {e}")
                finally:
                    # Update progress bar for each completed task
                    pbar.update(1)
    
    elapsed_time = time.time() - start_time
    print(f"\nProcessed {len(all_results)} questions in {elapsed_time:.2f} seconds")
    save_results_to_csv(all_results, output_csv)
    calculate_statistics(all_results)

# ================ VISUALIZATION FUNCTIONS ================

def generate_similarity_heatmap(results: List[Dict[str, Any]], output_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        metrics = [
            "tfidf_similarity_exp1", 
            "spacy_similarity_exp1",
            "sentence_transformer_similarity_exp1", 
            "word_embedding_similarity_exp1",
            "bertscore_f1_exp1",
            "jaccard_similarity_exp1",
            "entity_overlap_exp1"
        ]
        data = []
        for result in results:
            row = {"directory": result["directory"]}
            for metric in metrics:
                row[metric] = result.get(metric, None)
            data.append(row)
        df = pd.DataFrame(data)
        df.set_index("directory", inplace=True)
        df.columns = [col.replace("_exp1", "").replace("_similarity", "").replace("bertscore_f1", "BERTScore").title() for col in df.columns]
        plt.figure(figsize=(12, max(8, len(results) * 0.4)))
        sns.heatmap(df, annot=True, cmap="YlGnBu", linewidths=.5, fmt=".2f", cbar_kws={'label': 'Similarity Score'})
        plt.title("Similarity Scores Across Different Metrics and Directories")
        plt.tight_layout()
        plt.savefig(output_path)
    except Exception:
        pass

def generate_concept_coverage_chart(results: List[Dict[str, Any]], output_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
        data = []
        for result in results:
            if result.get("concept_coverage_percent_exp1") is not None:
                data.append({
                    "directory": result["directory"],
                    "coverage": result["concept_coverage_percent_exp1"],
                    "missing_count": len(result.get("missing_concepts_exp1", []))
                })
        df = pd.DataFrame(data)
        if df.empty:
            return
        df = df.sort_values("coverage")
        plt.figure(figsize=(14, max(8, len(data) * 0.5)))
        bars = plt.barh(df["directory"], df["coverage"], color="skyblue")
        for i, (_, row) in enumerate(df.iterrows()):
            plt.text(row["coverage"] + 1, i, f"{row['missing_count']} missing concepts", va='center')
        plt.xlabel("Concept Coverage (%)")
        plt.title("Concept Coverage by Directory")
        plt.axvline(x=df["coverage"].mean(), color='red', linestyle='--', label=f"Average: {df['coverage'].mean():.2f}%")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
    except Exception:
        pass

def generate_metric_comparison_chart(results: List[Dict[str, Any]], output_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        metrics = [
            ("TF-IDF", "tfidf_similarity_exp1"),
            ("spaCy", "spacy_similarity_exp1"),
            ("Sentence Transformer", "sentence_transformer_similarity_exp1"), 
            ("BERTScore", "bertscore_f1_exp1"),
            ("Jaccard", "jaccard_similarity_exp1")
        ]
        data = []
        for result in results:
            row = {"directory": result["directory"]}
            for label, metric in metrics:
                row[label] = result.get(metric, None)
            data.append(row)
        df = pd.DataFrame(data)
        if df.empty:
            return
        df_melted = pd.melt(df, id_vars=["directory"], value_vars=[m[0] for m in metrics], var_name="Metric", value_name="Similarity")
        plt.figure(figsize=(12, 8))
        sns.boxplot(x="Metric", y="Similarity", data=df_melted)
        sns.swarmplot(x="Metric", y="Similarity", data=df_melted, color=".25", size=4)
        plt.title("Comparison of Different Similarity Metrics")
        plt.xlabel("Metric")
        plt.ylabel("Similarity Score")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(output_path)
    except Exception:
        pass

# ================ CSV READING FUNCTION ================

def get_question_from_csv(csv_path: str, row_index: int) -> Optional[str]:
    """
    Read the CSV file and return the QuestionCount for the specified row index.
    
    Args:
        csv_path: Path to the CSV file
        row_index: Row index (1-based)
        
    Returns:
        QuestionCount as string or None if not found
    """
    try:
        df = pd.read_csv(csv_path)
        if row_index <= 0 or row_index > len(df):
            print(f"Error: Row index {row_index} is out of range (1-{len(df)})")
            return None
            
        # Convert to 0-based indexing for pandas
        question_count = df.iloc[row_index-1].get('QuestionCount')
        if pd.isna(question_count):
            print(f"Error: No QuestionCount found in row {row_index}")
            return None
            
        return str(int(question_count))
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

# ================ MAIN EXECUTION BLOCK ================

if __name__ == "__main__":
    import multiprocessing
    num_cpus = 1
    
    # Check if command-line argument is provided
    if len(sys.argv) > 1:
        try:
            row_index = int(sys.argv[1])
            csv_path = "biostars_filtered_questions.csv"
            
            print(f"Looking for question at row {row_index} in {csv_path}...")
            question_count = get_question_from_csv(csv_path, row_index)
           
            if question_count:
                print(f"Found question {question_count}, processing...")
                script_dir = os.path.dirname(os.path.abspath(__file__))
                question_dir = os.path.join(script_dir, "Questions", question_count)
                
                if os.path.isdir(question_dir):
                    result = process_question(question_dir)
                    if result:
                        all_results = [result]
                        
                        # Save results to JSON in the question directory
                        json_output_path = os.path.join(question_dir, "Step4-Metrics.json")
                        save_results_to_json(result, json_output_path)
                        
                        # Also save to CSV for broader analysis in the question directory
                        output_csv = os.path.join(question_dir, f"text_similarity_analysis.csv")
                        if not os.path.exists(output_csv):
                            save_results_to_csv(all_results, output_csv)
                            print(f"Analysis for question {question_count} saved to {output_csv}")
                            calculate_statistics(all_results)
                        else:
                            print(f"CSV file already exists at {output_csv}, skipping analysis")
                    else:
                        print(f"Failed to process question {question_count}")
                else:
                    print(f"Question directory not found: {question_dir}")
            else:
                print("Failed to get question from CSV")
        except ValueError:
            print("Invalid row index. Please provide a valid integer.")
    else:
        # Original functionality - process multiple questions
        print(f"Using {num_cpus} worker threads")
        process_multiple_questions("Questions", "text_similarity_analysis.csv", max_questions=100, num_workers=num_cpus)
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            print("Loading results from CSV for visualization...")
            results_df = pd.read_csv("text_similarity_analysis.csv")
            for col in results_df.columns:
                if col not in ['directory']:
                    results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
            results = results_df.to_dict('records')
            
            print("Generating visualizations...")
            os.makedirs("visualizations", exist_ok=True)
            
            with tqdm(total=3, desc="Creating charts", unit="chart") as pbar:
                # Create heatmap
                heatmap_metrics = [
                    "tfidf_similarity_exp1", 
                    "spacy_similarity_exp1",
                    "sentence_transformer_similarity_exp1", 
                    "word_embedding_similarity_exp1",
                    "bertscore_f1_exp1",
                    "jaccard_similarity_exp1",
                    "entity_overlap_exp1"
                ]
                
                heatmap_df = results_df[['directory'] + [m for m in heatmap_metrics if m in results_df.columns]]
                if not heatmap_df.empty and len(heatmap_df.columns) > 1:
                    heatmap_df = heatmap_df.set_index('directory')
                    heatmap_df.columns = [col.replace("_exp1", "").replace("_similarity", "").replace("bertscore_f1", "BERTScore").title() for col in heatmap_df.columns]
                    plt.figure(figsize=(12, max(8, len(results) * 0.4)))
                    sns.heatmap(heatmap_df, annot=True, cmap="YlGnBu", linewidths=.5, fmt=".2f", cbar_kws={'label': 'Similarity Score'})
                    plt.title("Similarity Scores Across Different Metrics and Directories")
                    plt.tight_layout()
                    plt.savefig("visualizations/similarity_heatmap.png")
                pbar.update(1)
                
                # Create concept coverage chart
                generate_concept_coverage_chart(results, "visualizations/concept_coverage.png")
                pbar.update(1)
                
                # Create metric comparison chart
                generate_metric_comparison_chart(results, "visualizations/metric_comparison.png")
                pbar.update(1)
                
            print("All visualizations generated successfully!")
        except Exception as e:
            print(f"Error during visualization: {e}")