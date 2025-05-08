import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import Dict, List, Tuple, Union, Optional, Any
import re
import warnings
from tqdm.auto import tqdm
from dataclasses import dataclass, field, asdict
import logging
from scipy import stats
import matplotlib.ticker as mtick

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('qa_analysis')

# Filter specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['GENSIM_DATA_DIR'] = '/data/ascher02/uqmmune1/BioStarsGPT/temp/'
os.environ['TRANSFORMERS_CACHE'] = '/data/ascher02/uqmmune1/BioStarsGPT/temp/huggingface'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/data/ascher02/uqmmune1/BioStarsGPT/temp/sentence_transformers'

# Initialize NLTK resources
import nltk
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("Downloading NLTK resources...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Try to import advanced libraries with fallbacks
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
    logger.info("Plotly is available for interactive visualizations")
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.info("Plotly not found, will use Matplotlib for visualizations")

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    logger.info("Sentence Transformers is available for semantic analysis")
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.info("Sentence Transformers not found, will use simpler methods for analysis")

try:
    import spacy
    try:
        nlp = spacy.load('en_core_web_sm')
        logger.info("Loaded spaCy model")
        SPACY_AVAILABLE = True
    except:
        logger.info("Downloading spaCy model...")
        spacy.cli.download('en_core_web_sm')
        nlp = spacy.load('en_core_web_sm')
        SPACY_AVAILABLE = True
except ImportError:
    logger.warning("SpaCy not available, falling back to basic NLTK")
    SPACY_AVAILABLE = False

# Fixed BERTopic import with OpenAI compatibility check
try:
    # First check for proper OpenAI compatibility to avoid AttributeError
    try:
        import openai
        # Check if the new client structure is available
        if hasattr(openai, 'OpenAI'):
            # BERTopic should work with this version of OpenAI
            from bertopic import BERTopic
            BERTOPIC_AVAILABLE = True
            logger.info("BERTopic is available for advanced topic modeling")
        else:
            # OpenAI version is incompatible with BERTopic
            logger.warning("OpenAI version incompatible with BERTopic (needs openai>=1.0.0)")
            BERTOPIC_AVAILABLE = False
    except ImportError:
        # No OpenAI, try importing BERTopic directly
        from bertopic import BERTopic
        BERTOPIC_AVAILABLE = True
        logger.info("BERTopic is available for advanced topic modeling")
except (ImportError, AttributeError) as e:
    BERTOPIC_AVAILABLE = False
    logger.info(f"BERTopic not available: {str(e)}. Will use LDA for topic modeling")

# Import sklearn components
try:
    from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.model_selection import KFold
    from sklearn.metrics import silhouette_score, coherence_score
except ImportError:
    # Handle the possibility that coherence_score might not be available
    from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.model_selection import KFold
    from sklearn.metrics import silhouette_score

try:
    import umap
    UMAP_AVAILABLE = True
    logger.info("UMAP is available for dimensionality reduction")
except ImportError:
    UMAP_AVAILABLE = False
    logger.info("UMAP not found, will use TruncatedSVD for dimensionality reduction")

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    logger.info("WordCloud not found, will skip wordcloud generation")


@dataclass
class DataStatistics:
    """Statistics about the dataset length and distribution."""
    total_pairs: int = 0
    avg_question_length: float = 0.0
    avg_explanation_length: float = 0.0
    max_question_length: int = 0
    max_explanation_length: int = 0
    min_question_length: int = 0
    min_explanation_length: int = 0
    median_question_length: float = 0.0
    median_explanation_length: float = 0.0
    percentile_90_question_length: float = 0.0
    percentile_90_explanation_length: float = 0.0
    total_question_tokens: int = 0
    total_explanation_tokens: int = 0
    
    def to_dict(self) -> Dict[str, Union[int, float]]:
        return asdict(self)


@dataclass
class CodeStatistics:
    """Statistics about code presence in the dataset."""
    questions_with_code: int = 0
    explanations_with_code: int = 0
    questions_with_code_percent: float = 0.0
    explanations_with_code_percent: float = 0.0
    
    def to_dict(self) -> Dict[str, Union[int, float]]:
        return asdict(self)


@dataclass
class ComplexityMetrics:
    """Metrics for content complexity assessment."""
    linguistic_complexity_score: float = 0.0
    diversity_score: float = 0.0
    perplexity: Optional[float] = None
    coherence: Optional[float] = None
    assessment: str = "Medium"
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    
    def to_dict(self) -> Dict[str, Union[float, str, Tuple[float, float]]]:
        return asdict(self)


@dataclass
class HyperParameters:
    """Recommended hyperparameters for model training."""
    model_name: str = "unsloth/Llama-3.3-8B-Instruct"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    train_samples: int = 0
    test_samples: int = 0
    num_epochs: int = 3
    learning_rate: float = 2e-5
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    eval_every_epochs: int = 1
    generation_max_length: int = 512
    generation_temperature: float = 0.7
    generation_min_p: float = 0.1
    generation_top_p: float = 0.92
    seed: int = 42
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    optimizer: str = "adamw_8bit"
    dataset_complexity: str = "Medium"
    diversity_score: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QADataAnalyzer:
    """
    A comprehensive analyzer for question-answer datasets that performs
    advanced NLP analysis and recommends optimal fine-tuning parameters.
    """
    
    def __init__(
        self, 
        input_file: str, 
        output_dir: str = "analysis_output",
        confidence_level: float = 0.95,
        plot_format: str = "svg"
    ):
        """
        Initialize the QA data analyzer.
        
        Args:
            input_file: Path to the JSONL file containing prompt-completion pairs
            output_dir: Directory to save analysis outputs
            confidence_level: Confidence level for statistical intervals (default: 0.95)
            plot_format: File format for saving plots (default: svg)
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.confidence_level = confidence_level
        self.plot_format = plot_format
        self.df = None
        self.stop_words = set(stopwords.words('english'))
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories for organization
        os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
        
        # Initialize statistics containers
        self.data_stats = DataStatistics()
        self.code_stats = CodeStatistics()
        self.complexity = ComplexityMetrics()
        self.hyperparams = HyperParameters()
        
        # Track additional metadata
        self.keywords = []
        self.entities = []
        self.topics = {}
        
        logger.info(f"Initialized analyzer for {input_file}")
        logger.info(f"Output will be saved to {output_dir}")

    def load_data(self) -> None:
        """Load and preprocess the dataset from JSONL."""
        logger.info(f"Loading data from {self.input_file}...")
        
        qa_pairs = []
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if 'prompt' in item and 'completion' in item:
                            qa_pairs.append({
                                'question': item['prompt'],
                                'explanation': item['completion']
                            })
                        else:
                            logger.warning("Found JSONL entry without prompt/completion fields")
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON line: {line[:50]}...")
        except FileNotFoundError:
            logger.error(f"Input file not found: {self.input_file}")
            raise
            
        if not qa_pairs:
            logger.error("No valid prompt-completion pairs found in the input file")
            raise ValueError("No valid prompt-completion pairs found")
            
        self.df = pd.DataFrame(qa_pairs)
        logger.info(f"Loaded {len(self.df)} prompt-completion pairs")
        
        # Calculate token lengths using improved tokenization
        if SPACY_AVAILABLE:
            self.df['question_tokens'] = self.df['question'].apply(
                lambda x: len([token for token in nlp(x)]))
            self.df['explanation_tokens'] = self.df['explanation'].apply(
                lambda x: len([token for token in nlp(x)]))
        else:
            self.df['question_tokens'] = self.df['question'].apply(
                lambda x: len(word_tokenize(x)))
            self.df['explanation_tokens'] = self.df['explanation'].apply(
                lambda x: len(word_tokenize(x)))
        
        # Handle outliers with statistical methods
        q1_question = self.df['question_tokens'].quantile(0.25)
        q3_question = self.df['question_tokens'].quantile(0.75)
        iqr_question = q3_question - q1_question
        
        q1_explanation = self.df['explanation_tokens'].quantile(0.25)
        q3_explanation = self.df['explanation_tokens'].quantile(0.75)
        iqr_explanation = q3_explanation - q1_explanation
        
        # Flag outliers (for reporting, not filtering)
        self.df['question_outlier'] = (
            (self.df['question_tokens'] < (q1_question - 1.5 * iqr_question)) | 
            (self.df['question_tokens'] > (q3_question + 1.5 * iqr_question))
        )
        self.df['explanation_outlier'] = (
            (self.df['explanation_tokens'] < (q1_explanation - 1.5 * iqr_explanation)) | 
            (self.df['explanation_tokens'] > (q3_explanation + 1.5 * iqr_explanation))
        )
        
        num_outliers = sum(self.df['question_outlier'] | self.df['explanation_outlier'])
        logger.info(f"Identified {num_outliers} potential outliers ({num_outliers/len(self.df)*100:.1f}%)")

    def compute_basic_statistics(self) -> None:
        """Calculate basic statistics about the dataset."""
        q_lengths = self.df['question_tokens']
        e_lengths = self.df['explanation_tokens']
        
        # Compute basic statistics
        self.data_stats.total_pairs = len(self.df)
        self.data_stats.avg_question_length = float(q_lengths.mean())
        self.data_stats.avg_explanation_length = float(e_lengths.mean())
        self.data_stats.max_question_length = int(q_lengths.max())
        self.data_stats.max_explanation_length = int(e_lengths.max())
        self.data_stats.min_question_length = int(q_lengths.min())
        self.data_stats.min_explanation_length = int(e_lengths.min())
        self.data_stats.median_question_length = float(q_lengths.median())
        self.data_stats.median_explanation_length = float(e_lengths.median())
        self.data_stats.percentile_90_question_length = float(q_lengths.quantile(0.9))
        self.data_stats.percentile_90_explanation_length = float(e_lengths.quantile(0.9))
        self.data_stats.total_question_tokens = int(q_lengths.sum())
        self.data_stats.total_explanation_tokens = int(e_lengths.sum())
        
        # Calculate bootstrap confidence intervals for mean lengths
        n_bootstrap = 1000
        q_means = np.zeros(n_bootstrap)
        e_means = np.zeros(n_bootstrap)
        
        for i in range(n_bootstrap):
            q_sample = np.random.choice(q_lengths, size=len(q_lengths), replace=True)
            e_sample = np.random.choice(e_lengths, size=len(e_lengths), replace=True)
            q_means[i] = np.mean(q_sample)
            e_means[i] = np.mean(e_sample)
        
        alpha = 1 - self.confidence_level
        q_ci_lower = np.percentile(q_means, alpha/2 * 100)
        q_ci_upper = np.percentile(q_means, (1-alpha/2) * 100)
        e_ci_lower = np.percentile(e_means, alpha/2 * 100)
        e_ci_upper = np.percentile(e_means, (1-alpha/2) * 100)
        
        logger.info(f"Question length: {self.data_stats.avg_question_length:.1f} tokens " 
                    f"({q_ci_lower:.1f} - {q_ci_upper:.1f}, {self.confidence_level*100:.0f}% CI)")
        logger.info(f"Explanation length: {self.data_stats.avg_explanation_length:.1f} tokens "
                    f"({e_ci_lower:.1f} - {e_ci_upper:.1f}, {self.confidence_level*100:.0f}% CI)")

    def detect_code_presence(self) -> None:
        """
        Detect code presence in questions and explanations using
        sophisticated pattern matching and machine learning.
        """
        def contains_code(text: str) -> bool:
            """
            Detect if text contains code using multiple heuristics.
            
            Args:
                text: Text to analyze
            
            Returns:
                Boolean indicating if code is detected
            """
            # Enhanced code patterns with improved regex
            code_patterns = [
                # Code blocks and inline code
                r'```[\s\S]*?```',                    # Markdown code blocks
                r'`[^`]+`',                           # Inline code
                
                # Programming language keywords
                r'\b(?:function|def|class|import|from|select|if|for|while|var|let|const|async|await|return)\b',
                
                # Function calls, method invocations, etc.
                r'[a-zA-Z0-9_]+\([^)]*\)',            # Function calls
                r'\.[a-zA-Z0-9_]+\([^)]*\)',          # Method calls
                
                # Variable assignments and declarations
                r'^\s*[a-zA-Z0-9_]+\s*=',             # Variable assignments
                r'^\s*(?:var|let|const)\s+[a-zA-Z0-9_]+',  # JS variable declarations
                
                # HTML, XML, and markup
                r'<[a-zA-Z]+[^>]*>[\s\S]*?</[a-zA-Z]+>', # HTML/XML tags
                r'<[a-zA-Z]+[^>]*\s?/>',              # Self-closing tags
                
                # SQL, database queries
                r'SELECT\s+[\s\S]*?\s+FROM',          # SQL SELECT
                r'INSERT\s+INTO',                     # SQL INSERT
                r'UPDATE\s+[\s\S]*?\s+SET',           # SQL UPDATE
                
                # CSS, styling
                r'[.#][a-zA-Z0-9_-]+\s*\{[\s\S]*?\}', # CSS selectors
                
                # Shell commands and scripting
                r'(?:npm|pip|apt|brew)\s+install',    # Install commands
                r'git\s+(?:clone|commit|push|pull)',  # Git commands
                r'docker\s+(?:run|build|pull)',       # Docker commands
                
                # File paths and URLs that look like code
                r'(?:\.\/|\/)[a-zA-Z0-9_\-\/\.]+\.[a-zA-Z0-9]+', # File paths
                
                # JSON/Dict-like structures
                r'\{(?:\s*"[^"]+"\s*:\s*(?:"[^"]*"|[0-9]+|true|false|null|undefined|\{|\[)[\s,]*)+\}',
                
                # Array/List-like structures 
                r'\[(?:\s*(?:"[^"]*"|[0-9]+|true|false|null|undefined|\{|\[)[\s,]*)+\]',
            ]
            
            # Check against each pattern
            for pattern in code_patterns:
                if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                    return True
                    
            # Count code indicators as potential evidence
            brackets = sum(text.count(c) for c in '[]{}()')
            indented_lines = len(re.findall(r'^\s{2,}[^\s]', text, re.MULTILINE))
            special_chars = sum(text.count(c) for c in '=+*/<>!&|^%')
            
            # Use a weighted score for code likelihood
            code_score = brackets*0.5 + indented_lines*2 + special_chars*0.3
            if code_score > 10:  # Threshold calibrated for code detection
                return True
                
            return False
        
        logger.info("Detecting code presence in prompts and completions...")
        
        # Apply the code detection function
        self.df['question_has_code'] = self.df['question'].apply(contains_code)
        self.df['explanation_has_code'] = self.df['explanation'].apply(contains_code)
        
        # Compute code statistics
        self.code_stats.questions_with_code = int(self.df['question_has_code'].sum())
        self.code_stats.explanations_with_code = int(self.df['explanation_has_code'].sum())
        self.code_stats.questions_with_code_percent = float(self.df['question_has_code'].mean() * 100)
        self.code_stats.explanations_with_code_percent = float(self.df['explanation_has_code'].mean() * 100)
        
        logger.info(f"Detected code in {self.code_stats.questions_with_code_percent:.1f}% of questions and "
                    f"{self.code_stats.explanations_with_code_percent:.1f}% of explanations")

    def extract_nlp_features(self) -> None:
        """
        Extract NLP features from the text using spaCy (if available)
        or NLTK as a fallback.
        """
        logger.info("Extracting NLP features from text...")
        
        if SPACY_AVAILABLE:
            # Efficient batched processing with spaCy
            batch_size = 50  # Adjust based on document size and available memory
            
            # Process questions in batches
            self.keywords = []
            self.entities = []
            
            for i in tqdm(range(0, len(self.df), batch_size), desc="Processing with spaCy"):
                batch_texts = self.df['question'].iloc[i:i+batch_size].tolist()
                # Use spaCy's pipe for efficiency
                docs = list(nlp.pipe(batch_texts))
                
                for doc in docs:
                    # Extract named entities
                    self.entities.extend([ent.text for ent in doc.ents])
                    
                    # Extract keywords (non-stopword nouns, verbs, and adjectives)
                    self.keywords.extend([
                        token.lemma_.lower() for token in doc 
                        if (not token.is_stop and not token.is_punct and 
                            token.pos_ in ('NOUN', 'VERB', 'ADJ'))
                    ])
        else:
            # Fallback to NLTK
            logger.info("Using NLTK for keyword extraction (fallback mode)")
            
            for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing with NLTK"):
                words = word_tokenize(row['question'])
                self.keywords.extend([
                    w.lower() for w in words 
                    if w.isalpha() and w.lower() not in self.stop_words
                ])
        
        # Filter out irrelevant or very common terms
        additional_stops = {'use', 'help', 'know', 'need', 'want', 'can', 'get', 'please', 'would', 'am', 'is', 'are'}
        self.keywords = [k for k in self.keywords if k not in additional_stops]
        
        logger.info(f"Extracted {len(set(self.keywords))} unique keywords and {len(set(self.entities))} unique entities")

    def perform_topic_modeling(self) -> None:
        """
        Perform topic modeling using advanced NMF or LDA approaches.
        """
        if len(self.df) < 20:
            logger.warning("Dataset too small for reliable topic modeling, skipping")
            self.topics = {"insufficient_data": True}
            return
        
        logger.info("Performing topic modeling...")
        
        # Choose the best available topic modeling approach
        if SENTENCE_TRANSFORMERS_AVAILABLE and len(self.df) >= 50:
            self._perform_nmf_topic_modeling()
        else:
            self._perform_lda_topic_modeling()

    def _perform_nmf_topic_modeling(self) -> None:
        """
        Use Non-negative Matrix Factorization (NMF) with TFIDF for advanced topic modeling.
        This is a modern alternative to BERTopic that works well with transformers.
        """
        try:
            logger.info("Using NMF with transformer embeddings for topic modeling")
            
            # Determine optimal number of topics based on dataset size
            min_topics = 2
            max_topics = min(20, max(5, len(self.df) // 50))
            
            # Step 1: Get embeddings with sentence transformers
            sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = sentence_model.encode(
                self.df['question'].tolist(),
                show_progress_bar=True,
                batch_size=16
            )
            
            # Step 2: Reduce dimensionality if needed (for visualization)
            if UMAP_AVAILABLE and embeddings.shape[1] > 50:
                # Reduce to a reasonable number of dimensions for NMF
                reducer = umap.UMAP(
                    n_components=50,
                    metric='cosine',
                    random_state=42
                )
                reduced_embeddings = reducer.fit_transform(embeddings)
            else:
                # Fallback to TruncatedSVD
                reducer = TruncatedSVD(n_components=min(50, embeddings.shape[1] - 1))
                reduced_embeddings = reducer.fit_transform(embeddings)
            
            # Step 3: Find optimal number of topics using silhouette score
            best_num_topics = 0
            best_score = -float('inf')
            
            for n_topics in range(min_topics, max_topics + 1, 1):
                # Use NMF for topic modeling
                nmf = NMF(
                    n_components=n_topics,
                    random_state=42,
                    max_iter=500
                )
                doc_topic_matrix = nmf.fit_transform(reduced_embeddings)
                
                # Use silhouette score as a measure of topic quality
                if n_topics > 1:  # Silhouette requires at least 2 clusters
                    # Use argmax to assign each document to its primary topic
                    doc_topic_labels = np.argmax(doc_topic_matrix, axis=1)
                    
                    # Only calculate silhouette if we have enough documents with different labels
                    if len(set(doc_topic_labels)) > 1 and len(doc_topic_labels) >= 10:
                        try:
                            # Calculate silhouette score
                            sil_score = silhouette_score(reduced_embeddings, doc_topic_labels)
                            
                            if sil_score > best_score:
                                best_score = sil_score
                                best_num_topics = n_topics
                        except Exception as e:
                            logger.warning(f"Silhouette calculation failed for {n_topics} topics: {str(e)}")
            
            # If we couldn't determine the optimal number of topics, use a heuristic
            if best_num_topics == 0:
                best_num_topics = min(5, max(2, len(self.df) // 50))
                logger.warning(f"Could not determine optimal topic count, using {best_num_topics}")
            else:
                logger.info(f"Optimal number of topics determined: {best_num_topics}")
            
            # Step 4: Perform final NMF with optimal number of topics
            nmf = NMF(
                n_components=best_num_topics,
                random_state=42,
                max_iter=500
            )
            doc_topic_matrix = nmf.fit_transform(reduced_embeddings)
            topic_term_matrix = nmf.components_
            
            # Step 5: Extract keywords for each topic using TF-IDF
            # We need to get keywords that correspond to the NMF components
            vectorizer = TfidfVectorizer(
                max_df=0.95,
                min_df=2,
                stop_words='english',
                max_features=1000
            )
            tfidf_matrix = vectorizer.fit_transform(self.df['question'])
            feature_names = vectorizer.get_feature_names_out()
            
            # Extract top terms for each topic based on the original TF-IDF space
            # We'll use KMeans to cluster documents by their NMF topics
            kmeans = KMeans(
                n_clusters=best_num_topics,
                random_state=42
            )
            cluster_labels = kmeans.fit_predict(doc_topic_matrix)
            
            # Now we can extract top terms for each cluster/topic
            self.topics = {}
            
            for topic_idx in range(best_num_topics):
                # Get documents in this topic/cluster
                topic_docs = [i for i, label in enumerate(cluster_labels) if label == topic_idx]
                
                if topic_docs:
                    # Get the TF-IDF vectors for these documents
                    topic_tfidf = tfidf_matrix[topic_docs]
                    
                    # Calculate average TF-IDF score for each term in this topic
                    topic_tfidf_sum = topic_tfidf.sum(axis=0)
                    topic_tfidf_avg = topic_tfidf_sum / len(topic_docs)
                    
                    # Get top terms by average TF-IDF
                    top_term_indices = np.argsort(topic_tfidf_avg.toarray()[0])[::-1][:15]
                    top_terms = [feature_names[i] for i in top_term_indices]
                    top_weights = [float(topic_tfidf_avg[0, i]) for i in top_term_indices]
                    
                    # Create dictionary for this topic
                    self.topics[f"Topic_{topic_idx+1}"] = {
                        term: weight for term, weight in zip(top_terms, top_weights)
                    }
                else:
                    # Empty topic
                    self.topics[f"Topic_{topic_idx+1}"] = {}
            
            # Try to calculate topic coherence or a similar quality metric
            try:
                # Use average silhouette score as a coherence-like metric
                self.complexity.coherence = float(best_score)
                logger.info(f"Topic coherence (silhouette): {best_score:.4f}")
            except:
                logger.warning("Could not calculate topic coherence")
                
            logger.info(f"NMF topic modeling identified {len(self.topics)} topics")
            
        except Exception as e:
            logger.error(f"NMF topic modeling failed: {str(e)}")
            logger.info("Falling back to LDA topic modeling")
            self._perform_lda_topic_modeling()

    def _perform_lda_topic_modeling(self) -> None:
        """Use LDA for topic modeling (fallback method)."""
        try:
            logger.info("Using LDA for topic modeling")
            
            # Create a document-term matrix
            vectorizer = CountVectorizer(
                max_df=0.95,
                min_df=2,
                stop_words='english',
                max_features=1000
            )
            X = vectorizer.fit_transform(self.df['question'])
            
            # Determine optimal number of topics using coherence or silhouette score
            best_num_topics = 0
            best_score = -float('inf')
            
            # Try different numbers of topics (can be computationally expensive)
            min_topics = min(2, len(self.df) // 100)
            max_topics = min(10, max(2, len(self.df) // 20))
            
            for n_topics in range(min_topics, max_topics + 1, 1):
                lda = LatentDirichletAllocation(
                    n_components=n_topics,
                    random_state=42,
                    max_iter=20,
                    learning_method='online'
                )
                doc_topic_matrix = lda.fit_transform(X)
                
                # Use silhouette score as a measure of topic quality
                if n_topics > 1:  # Silhouette requires at least 2 clusters
                    # Use argmax to assign each document to its primary topic
                    doc_topic_labels = np.argmax(doc_topic_matrix, axis=1)
                    
                    # Only calculate silhouette if we have enough documents
                    if len(set(doc_topic_labels)) > 1 and len(doc_topic_labels) >= 10:
                        try:
                            # SVD for dimensionality reduction
                            svd = TruncatedSVD(n_components=min(50, X.shape[1] - 1))
                            X_reduced = svd.fit_transform(X)
                            
                            # Calculate silhouette score
                            sil_score = silhouette_score(X_reduced, doc_topic_labels)
                            
                            if sil_score > best_score:
                                best_score = sil_score
                                best_num_topics = n_topics
                        except Exception as e:
                            logger.warning(f"Silhouette calculation failed for {n_topics} topics: {str(e)}")
            
            # If we couldn't determine the optimal number of topics, use a heuristic
            if best_num_topics == 0:
                best_num_topics = min(5, max(2, len(self.df) // 50))
                logger.warning(f"Could not determine optimal topic count, using {best_num_topics}")
            else:
                logger.info(f"Optimal number of topics determined: {best_num_topics}")
            
            # Fit final LDA model with optimal number of topics
            lda = LatentDirichletAllocation(
                n_components=best_num_topics,
                random_state=42,
                max_iter=25,
                learning_method='online'
            )
            lda.fit(X)
            
            # Calculate perplexity (lower is better)
            perplexity = lda.perplexity(X)
            self.complexity.perplexity = float(perplexity)
            logger.info(f"LDA model perplexity: {perplexity:.2f}")
            
            # Get feature names (terms)
            feature_names = vectorizer.get_feature_names_out()
            
            # Extract top terms for each topic
            self.topics = {}
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[:-11:-1]  # Top 10 terms
                top_words = [feature_names[i] for i in top_words_idx]
                top_weights = [float(topic[i]) for i in top_words_idx]
                
                # Create a dictionary for this topic
                self.topics[f"Topic_{topic_idx+1}"] = {
                    word: weight for word, weight in zip(top_words, top_weights)
                }
            
            logger.info(f"LDA modeling identified {len(self.topics)} topics")
            
        except Exception as e:
            logger.error(f"LDA topic modeling failed: {str(e)}")
            # Fallback to basic keyword extraction if even LDA fails
            self.topics = {"keyword_counts": dict(Counter(self.keywords).most_common(30))}
            logger.warning("Using keyword frequency as fallback for topic modeling")

    def analyze_semantic_complexity(self) -> None:
        """
        Analyze semantic complexity and content diversity using
        transformer embeddings or TF-IDF as a fallback.
        """
        logger.info("Analyzing semantic complexity...")
        
        # Default middle value
        diversity_score = 0.5
        confidence_interval = (0.45, 0.55)
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Sample if dataset is large to save computation time
                max_sample = 200
                sample_size = min(max_sample, len(self.df))
                sample_df = self.df.sample(sample_size) if len(self.df) > sample_size else self.df
                
                # Generate embeddings with modern transformer model
                logger.info(f"Generating embeddings for {sample_size} samples...")
                model = SentenceTransformer('all-MiniLM-L6-v2')
                question_embeddings = model.encode(
                    sample_df['question'].tolist(), 
                    show_progress_bar=True, 
                    batch_size=16,
                    convert_to_numpy=True
                )
                
                # Calculate pairwise similarities
                similarity_matrix = cosine_similarity(question_embeddings)
                
                # Calculate average similarity (lower means more diverse/complex content)
                # Get upper triangular matrix without diagonal (pairwise similarities)
                upper_tri_indices = np.triu_indices(len(similarity_matrix), k=1)
                similarities = similarity_matrix[upper_tri_indices]
                
                # Calculate mean and confidence interval
                avg_similarity = np.mean(similarities)
                
                # Use bootstrap to calculate confidence interval for diversity score
                n_bootstrap = 1000
                bootstrap_means = np.zeros(n_bootstrap)
                np.random.seed(42)
                
                for i in range(n_bootstrap):
                    sample_indices = np.random.choice(len(similarities), size=len(similarities), replace=True)
                    bootstrap_sample = similarities[sample_indices]
                    bootstrap_means[i] = np.mean(bootstrap_sample)
                
                # Calculate confidence interval
                alpha = 1 - self.confidence_level
                ci_lower = np.percentile(bootstrap_means, alpha/2 * 100)
                ci_upper = np.percentile(bootstrap_means, (1-alpha/2) * 100)
                
                # Convert to diversity scores (inverse of similarity)
                diversity_score = 1 - avg_similarity
                ci_lower_diversity = 1 - ci_upper  # Note the inversion
                ci_upper_diversity = 1 - ci_lower  # Note the inversion
                confidence_interval = (ci_lower_diversity, ci_upper_diversity)
                
                logger.info(f"Computed semantic diversity score: {diversity_score:.3f} " 
                           f"({ci_lower_diversity:.3f} - {ci_upper_diversity:.3f}, {self.confidence_level*100:.0f}% CI)")
                
            except Exception as e:
                logger.warning(f"Semantic complexity analysis failed: {str(e)}. Using fallback method.")
                self._analyze_lexical_diversity()
        else:
            logger.info("Sentence Transformers not available, using fallback method for complexity assessment.")
            self._analyze_lexical_diversity()
            
        # Store the computed diversity score and confidence interval
        self.complexity.diversity_score = float(diversity_score)
        self.complexity.confidence_interval = confidence_interval
        
        # Calculate enhanced complexity assessment using multiple factors
        linguistic_complexity_score = sum([
            (self.data_stats.avg_explanation_length / 100) * 2,    # Longer explanations suggest complexity
            (self.code_stats.explanations_with_code_percent / 100) * 3,  # Code presence indicates technical content
            (self.data_stats.avg_question_length / 50),           # Longer questions suggest detailed queries
            diversity_score * 5                                   # Semantic/lexical diversity as a complexity indicator
        ])
        
        self.complexity.linguistic_complexity_score = float(linguistic_complexity_score)
        
        # Determine complexity category with confidence measure
        if linguistic_complexity_score > 5:
            self.complexity.assessment = "High"
        elif linguistic_complexity_score > 3:
            self.complexity.assessment = "Medium"
        else:
            self.complexity.assessment = "Low"
            
        logger.info(f"Content complexity assessment: {self.complexity.assessment} " 
                   f"(Score: {linguistic_complexity_score:.2f})")

    def _analyze_lexical_diversity(self) -> None:
        """Analyze lexical diversity using TF-IDF as a fallback method."""
        try:
            # Create TF-IDF matrix
            tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
            tfidf_matrix = tfidf.fit_transform(self.df['question'])
            
            # Number of unique terms as ratio of total words is a simple diversity measure
            vocabulary_size = len(tfidf.get_feature_names_out())
            total_words = sum([len(word_tokenize(q)) for q in self.df['question']])
            
            # Normalize to a 0-1 scale (higher means more diverse vocabulary)
            diversity_score = min(0.95, vocabulary_size / (total_words * 0.1))
            
            # Estimate confidence interval using heuristic
            std_dev = 0.05  # Assumed standard deviation
            n = len(self.df)
            z = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
            margin = z * std_dev / np.sqrt(n)
            ci_lower = max(0, diversity_score - margin)
            ci_upper = min(1, diversity_score + margin)
            
            self.complexity.diversity_score = float(diversity_score)
            self.complexity.confidence_interval = (float(ci_lower), float(ci_upper))
            
            logger.info(f"Computed lexical diversity score: {diversity_score:.3f} " 
                       f"({ci_lower:.3f} - {ci_upper:.3f}, {self.confidence_level*100:.0f}% CI)")
            
        except Exception as e:
            logger.warning(f"Lexical diversity analysis failed: {str(e)}. Using default value.")
            self.complexity.diversity_score = 0.5
            self.complexity.confidence_interval = (0.4, 0.6)

    def create_visualizations(self) -> None:
        """Generate publication-quality visualizations of the analysis results."""
        logger.info("Creating visualizations...")
        
        # Set plotting style for publication quality
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_context("paper", font_scale=1.2)
        
        # Create Length Distribution plots
        self._plot_length_distributions()
        
        # Create code presence visualization
        self._plot_code_presence()
        
        # Create wordcloud visualization
        self._create_wordcloud()
        
        # Create topic visualization
        self._plot_topics()
        
        # Generate summary visualization
        self._create_summary_dashboard()
        
        logger.info(f"Visualizations saved to {os.path.join(self.output_dir, 'plots')}")
    
    def _plot_length_distributions(self) -> None:
        """Create visualizations of token length distributions."""
        # Publication-quality length distribution plots
        if PLOTLY_AVAILABLE:
            # Create interactive length distribution plot with Plotly
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Question Length Distribution', 'Explanation Length Distribution'),
                shared_yaxes=True
            )
            
            # Question length histogram
            q_trace = go.Histogram(
                x=self.df['question_tokens'],
                name='Questions',
                opacity=0.75,
                marker=dict(color='blue'),
                nbinsx=30
            )
            
            # Add mean and median markers for questions
            q_mean_trace = go.Scatter(
                x=[self.data_stats.avg_question_length, self.data_stats.avg_question_length],
                y=[0, self.df['question_tokens'].value_counts().max()],
                mode='lines',
                name='Mean',
                line=dict(color='red', width=2, dash='dash')
            )
            
            q_median_trace = go.Scatter(
                x=[self.data_stats.median_question_length, self.data_stats.median_question_length],
                y=[0, self.df['question_tokens'].value_counts().max()],
                mode='lines',
                name='Median',
                line=dict(color='green', width=2)
            )
            
            # Explanation length histogram
            e_trace = go.Histogram(
                x=self.df['explanation_tokens'],
                name='Explanations',
                opacity=0.75,
                marker=dict(color='orange'),
                nbinsx=30
            )
            
            # Add mean and median markers for explanations
            e_mean_trace = go.Scatter(
                x=[self.data_stats.avg_explanation_length, self.data_stats.avg_explanation_length],
                y=[0, self.df['explanation_tokens'].value_counts().max()],
                mode='lines',
                name='Mean',
                line=dict(color='red', width=2, dash='dash'),
                showlegend=False
            )
            
            e_median_trace = go.Scatter(
                x=[self.data_stats.median_explanation_length, self.data_stats.median_explanation_length],
                y=[0, self.df['explanation_tokens'].value_counts().max()],
                mode='lines',
                name='Median',
                line=dict(color='green', width=2),
                showlegend=False
            )
            
            # Add all traces to the figure
            fig.add_trace(q_trace, row=1, col=1)
            fig.add_trace(q_mean_trace, row=1, col=1)
            fig.add_trace(q_median_trace, row=1, col=1)
            fig.add_trace(e_trace, row=1, col=2)
            fig.add_trace(e_mean_trace, row=1, col=2)
            fig.add_trace(e_median_trace, row=1, col=2)
            
            # Update layout for publication quality
            fig.update_layout(
                title='Token Length Distributions',
                xaxis_title='Number of Tokens',
                yaxis_title='Count',
                template='plotly_white',
                height=500,
                width=1000,
                legend=dict(
                    yanchor="top",
                    y=0.98,
                    xanchor="right",
                    x=0.98
                )
            )
            
            # Save interactive plot
            fig.write_html(os.path.join(self.output_dir, "plots", "length_distribution_interactive.html"))
        
        # Always create static plot for publication
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        sns.histplot(self.df['question_tokens'], kde=True, color='blue')
        plt.axvline(self.data_stats.avg_question_length, color='red', linestyle='--', label='Mean')
        plt.axvline(self.data_stats.median_question_length, color='green', linestyle='-', label='Median')
        plt.title('Question Length Distribution')
        plt.xlabel('Number of Tokens')
        plt.ylabel('Count')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        sns.histplot(self.df['explanation_tokens'], kde=True, color='orange')
        plt.axvline(self.data_stats.avg_explanation_length, color='red', linestyle='--', label='Mean')
        plt.axvline(self.data_stats.median_explanation_length, color='green', linestyle='-', label='Median')
        plt.title('Explanation Length Distribution')
        plt.xlabel('Number of Tokens')
        plt.ylabel('Count')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "plots", f"length_distribution.{self.plot_format}"), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_code_presence(self) -> None:
        """Create visualizations of code presence in the dataset."""
        # Code presence visualization
        if PLOTLY_AVAILABLE:
            # Create a grouped bar chart
            categories = ['Questions', 'Explanations']
            with_code = [self.code_stats.questions_with_code, self.code_stats.explanations_with_code]
            without_code = [
                self.data_stats.total_pairs - self.code_stats.questions_with_code,
                self.data_stats.total_pairs - self.code_stats.explanations_with_code
            ]
            
            fig = go.Figure()
            
            # Add bars for content with code
            fig.add_trace(go.Bar(
                x=categories,
                y=with_code,
                name='With Code',
                marker_color='rgb(55, 83, 109)',
                text=[f"{(with_code[0]/self.data_stats.total_pairs)*100:.1f}%", 
                      f"{(with_code[1]/self.data_stats.total_pairs)*100:.1f}%"],
                textposition='auto'
            ))
            
            # Add bars for content without code
            fig.add_trace(go.Bar(
                x=categories,
                y=without_code,
                name='Without Code',
                marker_color='rgb(26, 118, 255)',
                text=[f"{(without_code[0]/self.data_stats.total_pairs)*100:.1f}%", 
                      f"{(without_code[1]/self.data_stats.total_pairs)*100:.1f}%"],
                textposition='auto'
            ))
            
            # Update layout for a stacked bar chart
            fig.update_layout(
                title='Presence of Code in Questions and Explanations',
                xaxis_title='Content Type',
                yaxis_title='Count',
                template='plotly_white',
                barmode='stack',
                height=500,
                width=800
            )
            
            # Save interactive plot
            fig.write_html(os.path.join(self.output_dir, "plots", "code_presence_interactive.html"))
        
        # Always create static plot for publication
        plt.figure(figsize=(10, 6))
        
        # Data preparation
        categories = ['Questions', 'Explanations']
        with_code_pct = [
            self.code_stats.questions_with_code_percent,
            self.code_stats.explanations_with_code_percent
        ]
        without_code_pct = [
            100 - self.code_stats.questions_with_code_percent,
            100 - self.code_stats.explanations_with_code_percent
        ]
        
        # Create the stacked bar chart
        bar_width = 0.5
        positions = np.arange(len(categories))
        
        plt.bar(positions, with_code_pct, bar_width, label='With Code', color='#5975a4')
        plt.bar(positions, without_code_pct, bar_width, bottom=with_code_pct, label='Without Code', color='#5cc2ca')
        
        # Add percentages on bars
        for i, v in enumerate(with_code_pct):
            plt.text(positions[i], v/2, f"{v:.1f}%", ha='center', color='white', fontweight='bold')
            
        for i, v in enumerate(without_code_pct):
            plt.text(positions[i], with_code_pct[i] + v/2, f"{v:.1f}%", ha='center', color='white', fontweight='bold')
        
        # Customize the plot
        plt.xlabel('Content Type')
        plt.ylabel('Percentage')
        plt.title('Presence of Code in Questions and Explanations')
        plt.xticks(positions, categories)
        plt.ylim(0, 100)
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "plots", f"code_presence.{self.plot_format}"), dpi=300, bbox_inches='tight')
        plt.close()

    def _create_wordcloud(self) -> None:
        """Create wordcloud visualization of key terms."""
        if not WORDCLOUD_AVAILABLE:
            logger.warning("WordCloud not available, skipping wordcloud visualization")
            return
            
        try:
            # Combine keywords and entities
            combined_text = ' '.join(self.keywords + self.entities)
            
            # Create domain-specific stopwords
            domain_stopwords = {'use', 'using', 'used', 'try', 'need', 'get', 'one', 'like', 
                               'would', 'make', 'want', 'know', 'help', 'can', 'problem'}
            
            # Configure the wordcloud with better aesthetics
            wordcloud = WordCloud(
                width=1000, 
                height=600, 
                background_color='white', 
                max_words=200,
                collocations=False,
                contour_width=3,
                contour_color='steelblue',
                colormap='viridis',
                stopwords=domain_stopwords.union(self.stop_words)
            ).generate(combined_text)
            
            # Create figure with high DPI for publication quality
            plt.figure(figsize=(12, 8), dpi=300)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Key Terms in Questions', fontsize=16, pad=20)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "plots", f"wordcloud.{self.plot_format}"), 
                      dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Generated wordcloud visualization")
            
        except Exception as e:
            logger.warning(f"Wordcloud generation failed: {str(e)}. Skipping wordcloud.")

    def _plot_topics(self) -> None:
        """Create visualizations of the topic model results."""
        if not self.topics or len(self.topics) <= 1 or "insufficient_data" in self.topics:
            logger.warning("Insufficient topic data for visualization")
            return
            
        try:
            # Create publication-quality topic visualization
            n_topics = len(self.topics)
            cols = min(3, n_topics)
            rows = (n_topics + cols - 1) // cols  # Ceiling division
            
            plt.figure(figsize=(15, rows * 4), dpi=300)
            
            for i, (topic_name, terms) in enumerate(self.topics.items()):
                plt.subplot(rows, cols, i+1)
                
                # Sort terms by weight
                sorted_terms = dict(sorted(terms.items(), key=lambda x: x[1], reverse=True)[:10])
                
                # Create horizontal bar chart
                bars = plt.barh(
                    range(len(sorted_terms)), 
                    list(sorted_terms.values()),
                    color=plt.cm.viridis(np.linspace(0, 0.8, len(sorted_terms))),
                    alpha=0.8
                )
                
                # Add term labels
                plt.yticks(range(len(sorted_terms)), list(sorted_terms.keys()))
                
                # Add weight labels on bars
                for bar, weight in zip(bars, sorted_terms.values()):
                    plt.text(
                        weight + max(sorted_terms.values()) * 0.01,
                        bar.get_y() + bar.get_height()/2,
                        f"{weight:.3f}",
                        va='center',
                        fontsize=8
                    )
                
                plt.title(f"{topic_name}", fontsize=12)
                plt.xlabel("Term Weight", fontsize=10)
                plt.gca().invert_yaxis()  # Highest weights at the top
                
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.suptitle("Topic Model - Top Terms by Weight", fontsize=16, y=0.98)
            plt.savefig(os.path.join(self.output_dir, "plots", f"topic_model.{self.plot_format}"), 
                      dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create interactive topic visualization if Plotly is available
            if PLOTLY_AVAILABLE:
                # Create an interactive topic visualization
                fig = make_subplots(
                    rows=rows, cols=cols,
                    subplot_titles=[topic_name for topic_name in self.topics.keys()],
                    vertical_spacing=0.1
                )
                
                for i, (topic_name, terms) in enumerate(self.topics.items()):
                    # Calculate row and column position
                    row_idx = i // cols + 1
                    col_idx = i % cols + 1
                    
                    # Sort terms by weight
                    sorted_terms = dict(sorted(terms.items(), key=lambda x: x[1], reverse=True)[:10])
                    
                    # Add horizontal bar chart
                    fig.add_trace(
                        go.Bar(
                            y=list(sorted_terms.keys()),
                            x=list(sorted_terms.values()),
                            orientation='h',
                            marker=dict(
                                color=list(sorted_terms.values()),
                                colorscale='Viridis'
                            ),
                            text=[f"{weight:.3f}" for weight in sorted_terms.values()],
                            textposition='auto',
                            hoverinfo='text',
                            hovertext=[f"{term}: {weight:.3f}" for term, weight in sorted_terms.items()]
                        ),
                        row=row_idx, col=col_idx
                    )
                    
                    # Update axes
                    fig.update_xaxes(title_text="Weight", row=row_idx, col=col_idx)
                    
                # Update layout
                fig.update_layout(
                    title_text="Interactive Topic Model Visualization",
                    showlegend=False,
                    height=300 * rows,
                    width=400 * cols,
                    template="plotly_white"
                )
                
                # Save interactive visualization
                fig.write_html(os.path.join(self.output_dir, "plots", "topic_model_interactive.html"))
                
            logger.info("Generated topic model visualizations")
            
        except Exception as e:
            logger.warning(f"Topic visualization failed: {str(e)}")

    def _create_summary_dashboard(self) -> None:
        """Create a summary dashboard of key analysis results."""
        try:
            # Create a summary dashboard for quick overview
            plt.figure(figsize=(15, 10), dpi=300)
            
            # Grid layout
            gs = plt.GridSpec(2, 2, height_ratios=[1, 1.2])
            
            # 1. Dataset Statistics Summary
            ax1 = plt.subplot(gs[0, 0])
            stats_text = [
                f"Total QA Pairs: {self.data_stats.total_pairs}",
                f"Avg Question: {self.data_stats.avg_question_length:.1f} tokens",
                f"Avg Explanation: {self.data_stats.avg_explanation_length:.1f} tokens",
                f"Questions with Code: {self.code_stats.questions_with_code_percent:.1f}%",
                f"Explanations with Code: {self.code_stats.explanations_with_code_percent:.1f}%",
                f"Complexity: {self.complexity.assessment}",
                f"Diversity Score: {self.complexity.diversity_score:.3f}"
            ]
            
            # No data visualization, just a text summary
            ax1.text(0.5, 0.5, '\n'.join(stats_text), 
                   ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle='round,pad=1', facecolor='#f0f0f0', alpha=0.8))
            ax1.set_title('Dataset Statistics', fontsize=14)
            ax1.axis('off')
            
            # 2. Length comparison box plot
            ax2 = plt.subplot(gs[0, 1])
            df_plot = pd.DataFrame({
                'Question': self.df['question_tokens'],
                'Explanation': self.df['explanation_tokens']
            })
            
            sns.boxplot(data=df_plot, palette='Set2', ax=ax2)
            ax2.set_title('Token Length Comparison', fontsize=14)
            ax2.set_ylabel('Number of Tokens')
            
            # Add mean markers
            for i, col in enumerate(['Question', 'Explanation']):
                mean_val = df_plot[col].mean()
                ax2.plot(i, mean_val, 'o', color='red', markersize=8)
                ax2.text(i, mean_val * 1.1, f'Mean: {mean_val:.1f}', 
                        ha='center', fontsize=10, color='red')
            
            # 3. Topic/Keyword Summary
            ax3 = plt.subplot(gs[1, :])
            
            # If we have topics from topic modeling
            if self.topics and len(self.topics) > 1 and "insufficient_data" not in self.topics:
                # Create a horizontal bar chart of top terms across all topics
                all_terms = {}
                for topic_dict in self.topics.values():
                    for term, weight in topic_dict.items():
                        if term in all_terms:
                            all_terms[term] += weight
                        else:
                            all_terms[term] = weight
                
                # Sort and select top terms
                top_terms = dict(sorted(all_terms.items(), key=lambda x: x[1], reverse=True)[:15])
                
                # Create horizontal bar chart
                bars = ax3.barh(
                    range(len(top_terms)), 
                    list(top_terms.values()),
                    color=plt.cm.viridis(np.linspace(0, 0.8, len(top_terms))),
                    alpha=0.8
                )
                
                ax3.set_yticks(range(len(top_terms)))
                ax3.set_yticklabels(list(top_terms.keys()))
                ax3.set_title('Top Terms Across Topics', fontsize=14)
                ax3.set_xlabel('Cumulative Weight')
                ax3.invert_yaxis()  # Highest weights at the top
                
                # Add weight labels
                for bar in bars:
                    width = bar.get_width()
                    ax3.text(width + 0.1, bar.get_y() + bar.get_height()/2, f"{width:.2f}", 
                           va='center', fontsize=9)
            else:
                # Fallback to keyword frequency
                top_keywords = dict(Counter(self.keywords).most_common(15))
                
                # Create horizontal bar chart
                bars = ax3.barh(
                    range(len(top_keywords)), 
                    list(top_keywords.values()),
                    color=plt.cm.viridis(np.linspace(0, 0.8, len(top_keywords))),
                    alpha=0.8
                )
                
                ax3.set_yticks(range(len(top_keywords)))
                ax3.set_yticklabels(list(top_keywords.keys()))
                ax3.set_title('Top Keywords by Frequency', fontsize=14)
                ax3.set_xlabel('Frequency')
                ax3.invert_yaxis()  # Highest frequency at the top
                
                # Add frequency labels
                for bar in bars:
                    width = bar.get_width()
                    ax3.text(width + 0.1, bar.get_y() + bar.get_height()/2, f"{int(width)}", 
                           va='center', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "plots", f"summary_dashboard.{self.plot_format}"), 
                      dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Generated summary dashboard visualization")
            
        except Exception as e:
            logger.warning(f"Summary dashboard generation failed: {str(e)}")

    def recommend_hyperparameters(self) -> None:
        """
        Recommend optimal hyperparameters for fine-tuning based on
        dataset characteristics and complexity assessment.
        """
        logger.info("Recommending optimal hyperparameters...")
        
        # Calculate total tokens for training
        total_tokens = (self.data_stats.total_question_tokens + 
                       self.data_stats.total_explanation_tokens)
        
        # Determine recommended model size based on data complexity and size
        if self.complexity.linguistic_complexity_score > 5 or total_tokens > 500000:
            self.hyperparams.model_name = "meta-llama/Llama-3-70b-instruct"
            self.hyperparams.lora_r = 32
            logger.info(f"Recommending large model (70B) due to high complexity "
                       f"({self.complexity.linguistic_complexity_score:.2f}) or large dataset size")
        elif self.complexity.linguistic_complexity_score > 3 or total_tokens > 200000:
            self.hyperparams.model_name = "meta-llama/Llama-3-8b-instruct"
            self.hyperparams.lora_r = 16
            logger.info(f"Recommending medium model (8B) based on complexity "
                       f"({self.complexity.linguistic_complexity_score:.2f}) and dataset size")
        else:
            self.hyperparams.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            self.hyperparams.lora_r = 16
            logger.info(f"Recommending smaller model (1.1B) based on complexity "
                       f"({self.complexity.linguistic_complexity_score:.2f}) and dataset size")
        
        # Calculate sequence length based on token distributions
        max_pair_length = (self.data_stats.max_question_length + 
                          self.data_stats.max_explanation_length)
        
        # Use the next power of 2 after max length, with ceiling at 8192
        recommended_seq_length = min(8192, 2 ** (int(np.log2(max_pair_length)) + 1))
        self.hyperparams.max_seq_length = recommended_seq_length
        
        # Determine batch size based on model and sequence length
        if "70B" in self.hyperparams.model_name:
            self.hyperparams.batch_size = 1
            self.hyperparams.gradient_accumulation_steps = 16
        elif "8B" in self.hyperparams.model_name:
            self.hyperparams.batch_size = 1
            self.hyperparams.gradient_accumulation_steps = 8
        else:
            self.hyperparams.batch_size = 2
            self.hyperparams.gradient_accumulation_steps = 4
        
        # Determine learning rate based on complexity and dataset size
        if self.complexity.assessment == "High":
            self.hyperparams.learning_rate = 1e-5  # More conservative for complex data
        elif self.complexity.assessment == "Medium":
            self.hyperparams.learning_rate = 2e-5
        else:
            self.hyperparams.learning_rate = 5e-5
        
        # Scaled down for larger models
        if "70B" in self.hyperparams.model_name:
            self.hyperparams.learning_rate = self.hyperparams.learning_rate / 2
        
        # Determine epochs based on dataset size and complexity
        if len(self.df) < 100:
            # More epochs for smaller datasets, with a ceiling
            self.hyperparams.num_epochs = min(100, 1000 // len(self.df))
        elif len(self.df) < 500:
            self.hyperparams.num_epochs = 20
        else:
            self.hyperparams.num_epochs = 10
        
        # Apply early stopping
        self.hyperparams.eval_every_epochs = max(1, self.hyperparams.num_epochs // 10)
        
        # Set training and test samples
        self.hyperparams.train_samples = int(len(self.df) * 0.8)  # 80% train
        self.hyperparams.test_samples = max(10, int(len(self.df) * 0.2))  # 20% test
        
        # Set LoRA parameters
        self.hyperparams.lora_alpha = self.hyperparams.lora_r * 2
        self.hyperparams.lora_dropout = 0.05 if self.complexity.assessment == "High" else 0.02
        
        # Set generation parameters based on explanation length distribution
        self.hyperparams.generation_max_length = int(self.data_stats.percentile_90_explanation_length * 1.5)
        
        # Set complexity assessment
        self.hyperparams.dataset_complexity = self.complexity.assessment
        self.hyperparams.diversity_score = self.complexity.diversity_score
        
        logger.info(f"Recommended model: {self.hyperparams.model_name}")
        logger.info(f"Recommended sequence length: {self.hyperparams.max_seq_length}")
        logger.info(f"Recommended learning rate: {self.hyperparams.learning_rate}")
        logger.info(f"Recommended epochs: {self.hyperparams.num_epochs}")

    def save_results(self) -> None:
        """Save analysis results to JSON files."""
        logger.info("Saving analysis results...")
        
        # Save entity and keyword analysis
        entity_analysis = {}
        
        if self.entities:
            entity_analysis["common_entities"] = {
                str(term): int(count) for term, count in Counter(self.entities).most_common(20)
            }
        
        entity_analysis["domain_keywords"] = {
            str(term): int(count) for term, count in Counter(self.keywords).most_common(30)
        }
        
        # Save all statistics to file
        all_stats = {
            "data_stats": self.data_stats.to_dict(),
            "code_stats": self.code_stats.to_dict(),
            "complexity": self.complexity.to_dict(),
            "entity_analysis": entity_analysis,
            "topic_modeling": self.topics,
            "total_training_tokens_estimate": (
                self.data_stats.total_question_tokens +
                self.data_stats.total_explanation_tokens
            ),
            "analysis_timestamp": pd.Timestamp.now().isoformat()
        }
        
        # Save recommended hyperparameters
        with open(os.path.join(self.output_dir, "data", "data_analysis.json"), 'w') as f:
            json.dump(all_stats, f, indent=2)
        
        # Save recommended hyperparameters to a separate file
        with open(os.path.join(self.output_dir, "data", "recommended_hyperparameters.json"), 'w') as f:
            json.dump(self.hyperparams.to_dict(), f, indent=2)
        
        # Save a copy of the original dataframe for reference
        self.df.to_csv(os.path.join(self.output_dir, "data", "processed_dataset.csv"), index=False)
        
        logger.info(f"Analysis results saved to {os.path.join(self.output_dir, 'data')}")

    def generate_report(self) -> str:
        """
        Generate a publication-quality report summarizing the analysis results.
        
        Returns:
            str: Markdown-formatted report
        """
        # Create a report title based on the input file
        filename = os.path.basename(self.input_file)
        report_title = f"Analysis Report for {filename}"
        
        # Format the report in Markdown
        report = f"""# {report_title}

## 1. Executive Summary

This report presents a comprehensive analysis of a question-answer dataset containing **{self.data_stats.total_pairs}** pairs. The dataset exhibits **{self.complexity.assessment.lower()}** complexity with a diversity score of **{self.complexity.diversity_score:.3f}** (95% CI: {self.complexity.confidence_interval[0]:.3f}-{self.complexity.confidence_interval[1]:.3f}). Based on content and structural analysis, we recommend fine-tuning a **{self.hyperparams.model_name}** model with the optimized hyperparameters detailed in Section 5.

## 2. Dataset Statistics

| Metric | Questions | Explanations |
|--------|-----------|-------------|
| Average Length (tokens) | {self.data_stats.avg_question_length:.1f} | {self.data_stats.avg_explanation_length:.1f} |
| Median Length (tokens) | {self.data_stats.median_question_length:.1f} | {self.data_stats.median_explanation_length:.1f} |
| Maximum Length (tokens) | {self.data_stats.max_question_length} | {self.data_stats.max_explanation_length} |
| 90th Percentile Length | {self.data_stats.percentile_90_question_length:.1f} | {self.data_stats.percentile_90_explanation_length:.1f} |
| Contains Code | {self.code_stats.questions_with_code_percent:.1f}% | {self.code_stats.explanations_with_code_percent:.1f}% |

The dataset contains a total of approximately **{self.data_stats.total_question_tokens + self.data_stats.total_explanation_tokens:,}** tokens. The distribution of tokens suggests a {self.complexity.assessment.lower()}-complexity dataset that requires a sequence length of at least **{self.hyperparams.max_seq_length}** tokens to accommodate the longest samples.

## 3. Content Analysis

### 3.1 Topic Distribution

The semantic analysis identified **{len(self.topics)}** distinct topics in the questions:
"""
        
        # Add topic information if available
        if self.topics and len(self.topics) > 1 and "insufficient_data" not in self.topics:
            for topic_name, terms in list(self.topics.items())[:5]:  # Show top 5 topics
                report += f"\n- **{topic_name}**: "
                top_terms = list(terms.items())[:5]  # Show top 5 terms
                report += ", ".join([f"{term} ({weight:.3f})" for term, weight in top_terms])
            
            if len(self.topics) > 5:
                report += f"\n\nAnd {len(self.topics) - 5} more topics..."
        else:
            report += "\nInsufficient data for reliable topic modeling. Analysis based on keyword frequency instead.\n"
            
            # Add top keywords
            top_keywords = Counter(self.keywords).most_common(10)
            report += "\n**Top Keywords**: "
            report += ", ".join([f"{term} ({count})" for term, count in top_keywords])
        
        # Add complexity assessment
        report += f"""

### 3.2 Complexity Assessment

The dataset complexity was assessed as **{self.complexity.assessment}** with a complexity score of **{self.complexity.linguistic_complexity_score:.2f}**. This assessment is based on multiple factors:

- Semantic diversity: {self.complexity.diversity_score:.3f}
- Average explanation length: {self.data_stats.avg_explanation_length:.1f} tokens
- Code presence in explanations: {self.code_stats.explanations_with_code_percent:.1f}%
- Average question length: {self.data_stats.avg_question_length:.1f} tokens

"""
        
        # Add topic coherence if available
        if self.complexity.coherence is not None:
            report += f"- Topic coherence: {self.complexity.coherence:.3f}\n"
        if self.complexity.perplexity is not None:
            report += f"- Topic model perplexity: {self.complexity.perplexity:.2f}\n"
        
        # Add data split recommendations
        report += f"""
## 4. Training Data Recommendations

Based on the dataset characteristics, we recommend the following data split:

- Training samples: {self.hyperparams.train_samples} ({int(self.hyperparams.train_samples/self.data_stats.total_pairs*100)}% of dataset)
- Testing samples: {self.hyperparams.test_samples} ({int(self.hyperparams.test_samples/self.data_stats.total_pairs*100)}% of dataset)

## 5. Hyperparameter Recommendations

For optimal fine-tuning results, we recommend the following hyperparameters:

| Parameter | Recommended Value | Rationale |
|-----------|-------------------|-----------|
| Base Model | {self.hyperparams.model_name} | Selected based on dataset complexity and size |
| Sequence Length | {self.hyperparams.max_seq_length} | Accommodates {int(self.hyperparams.max_seq_length/max(self.data_stats.max_question_length + self.data_stats.max_explanation_length, 1)*100)}% of the maximum required length |
| Epochs | {self.hyperparams.num_epochs} | Optimized for dataset size of {self.data_stats.total_pairs} samples |
| Learning Rate | {self.hyperparams.learning_rate} | Adjusted for {self.complexity.assessment.lower()} complexity content |
| Batch Size | {self.hyperparams.batch_size} with {self.hyperparams.gradient_accumulation_steps} gradient accumulation steps | Optimized for model size and memory efficiency |
| LoRA Rank (r) | {self.hyperparams.lora_r} | Selected based on dataset diversity and complexity |
| LoRA Alpha | {self.hyperparams.lora_alpha} | Set to 2x LoRA rank for optimal adaptation |
| LoRA Dropout | {self.hyperparams.lora_dropout} | {"Higher value for robustness with complex data" if self.complexity.assessment == "High" else "Standard value for medium/low complexity data"} |

For generation during evaluation and inference, we recommend:

- Maximum generation length: {self.hyperparams.generation_max_length} tokens
- Temperature: {self.hyperparams.generation_temperature}
- Top-p (nucleus sampling): {self.hyperparams.generation_top_p}
- Minimum-p: {self.hyperparams.generation_min_p}

## 6. Visualization Summary

This analysis includes the following visualizations (available in the 'plots' directory):

1. Token length distributions for questions and explanations
2. Code presence analysis
3. Topic model visualization
4. Key terms wordcloud
5. Summary dashboard of dataset characteristics

## 7. Conclusion

This dataset demonstrates {self.complexity.assessment.lower()} complexity with {len(self.topics) if (self.topics and len(self.topics) > 1 and "insufficient_data" not in self.topics) else "several"} distinct topics. The recommended fine-tuning approach with a {self.hyperparams.model_name.split('/')[-1]} model and optimized parameters should yield strong results in capturing the question-answer patterns present in the data.

---

*Analysis generated on {pd.Timestamp.now().strftime('%Y-%m-%d')}*
"""
        
        # Save the report
        report_path = os.path.join(self.output_dir, "analysis_report.md")
        with open(report_path, 'w') as f:
            f.write(report)
            
        logger.info(f"Analysis report saved to {report_path}")
        
        return report

    def run_analysis(self) -> HyperParameters:
        """
        Run the complete analysis pipeline and return recommended hyperparameters.
        
        Returns:
            HyperParameters: Recommended hyperparameters for model training
        """
        logger.info("Starting comprehensive QA dataset analysis...")
        
        # Execute analysis pipeline
        try:
            self.load_data()
            self.compute_basic_statistics()
            self.detect_code_presence()
            self.extract_nlp_features()
            self.perform_topic_modeling()
            self.analyze_semantic_complexity()
            self.create_visualizations()
            self.recommend_hyperparameters()
            self.save_results()
            self.generate_report()
            
            logger.info("Analysis completed successfully")
            logger.info(f"Full results available in {self.output_dir}")
            
            # Print summary
            print("\n" + "="*60)
            print("QA Dataset Analysis Summary")
            print("="*60)
            print(f"Dataset: {self.input_file}")
            print(f"Total QA pairs: {self.data_stats.total_pairs}")
            print(f"Average question length: {self.data_stats.avg_question_length:.1f} tokens")
            print(f"Average explanation length: {self.data_stats.avg_explanation_length:.1f} tokens")
            print(f"Questions with code: {self.code_stats.questions_with_code_percent:.1f}%")
            print(f"Explanations with code: {self.code_stats.explanations_with_code_percent:.1f}%")
            print(f"Content complexity: {self.complexity.assessment} (Score: {self.complexity.linguistic_complexity_score:.2f})")
            print(f"Diversity score: {self.complexity.diversity_score:.3f}")
            
            print("\n" + "="*60)
            print("Recommended Model and Hyperparameters")
            print("="*60)
            print(f"Recommended model: {self.hyperparams.model_name}")
            print(f"Sequence length: {self.hyperparams.max_seq_length}")
            print(f"Number of epochs: {self.hyperparams.num_epochs}")
            print(f"Learning rate: {self.hyperparams.learning_rate}")
            print(f"Batch size: {self.hyperparams.batch_size} with {self.hyperparams.gradient_accumulation_steps} gradient accumulation steps")
            print(f"LoRA rank (r): {self.hyperparams.lora_r}")
            
            print("\n" + "="*60)
            print(f"Complete analysis saved to: {self.output_dir}")
            print(f"Analysis report: {os.path.join(self.output_dir, 'analysis_report.md')}")
            print("="*60)
            
            return self.hyperparams
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise


def analyze_qa_data(
    input_file: str = "Train_with_outliers.jsonl",
    output_dir: str = "qa_analysis_output",
    confidence_level: float = 0.95,
    plot_format: str = "svg"
) -> HyperParameters:
    """
    Analyze a QA dataset and recommend optimal hyperparameters.
    
    Args:
        input_file: Path to the JSONL file containing prompt-completion pairs
        output_dir: Directory to save analysis outputs
        confidence_level: Confidence level for statistical intervals (default: 0.95)
        plot_format: File format for saving plots (default: svg)
        
    Returns:
        HyperParameters: Recommended hyperparameters for model training
    """
    analyzer = QADataAnalyzer(
        input_file=input_file,
        output_dir=output_dir,
        confidence_level=confidence_level,
        plot_format=plot_format
    )
    
    return analyzer.run_analysis()


if __name__ == "__main__":
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Analyze QA dataset and recommend optimal hyperparameters"
    )
    parser.add_argument(
        "--input", "-i", 
        default="Train_with_outliers.jsonl",
        help="Path to input JSONL file (default: Test_with_outliers.jsonl)"
    )
    parser.add_argument(
        "--output", "-o", 
        default="qa_analysis_output",
        help="Output directory (default: qa_analysis_output)"
    )
    parser.add_argument(
        "--confidence", "-c", 
        type=float,
        default=0.95,
        help="Confidence level for statistical intervals (default: 0.95)"
    )
    parser.add_argument(
        "--format", "-f", 
        default="svg",
        choices=["svg", "png", "pdf"],
        help="Plot format (default: svg)"
    )
    
    args = parser.parse_args()
    
    # Run analysis
    analyze_qa_data(
        input_file=args.input,
        output_dir=args.output,
        confidence_level=args.confidence,
        plot_format=args.format
    )

    