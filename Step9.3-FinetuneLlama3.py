from unsloth import FastLanguageModel
import torch
import json
import os
import numpy as np
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq, TextStreamer
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter
import csv
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
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
# Setup paths
# os.environ['GENSIM_DATA_DIR'] = '/data/ascher02/uqmmune1/BioStarsGPT/temp/'
# os.environ['TRANSFORMERS_CACHE'] = '/data/ascher02/uqmmune1/BioStarsGPT/temp/huggingface'
# os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/data/ascher02/uqmmune1/BioStarsGPT/temp/sentence_transformers'
# # Set environment variables
# os.environ["TRITON_CACHE_DIR"] = "/data/ascher02/uqmmune1/BioStarsGPT/temp/.triton_cache"
# os.environ["HF_HOME"] = "/data/ascher02/uqmmune1/BioStarsGPT/temp/.huggingface_cache"
# os.environ["TORCH_HOME"] = "/data/ascher02/uqmmune1/BioStarsGPT/temp/.torch_cache"
import nltk
from collections import defaultdict
 
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TextStreamer, 
    RobertaTokenizer, RobertaModel, AutoModel
)

# Set up NLTK
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab')

# Initialize availability flags
SPACY_AVAILABLE = False
BERT_AVAILABLE = False
WMD_AVAILABLE = False
NLI_AVAILABLE = False
BERTSCORE_AVAILABLE = False
CODEBERT_AVAILABLE = False


# ====== CONFIGURATION ======
train_samples = 144137  # Number of training samples to use
validation_samples = 10001  # Number of validation samples to use
test_samples = 100     # Number of test samples to use

# train_samples = 100  # Number of training samples to use
# validation_samples = 100  # Number of validation samples to use
# test_samples = 10     # Number of test samples to use


max_seq_length = 2048  # Context length for the model
dtype = None
load_in_4bit = True
model_name = "unsloth/Llama-3.2-3B-Instruct"
#model_name = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
num_epochs = 50  # Reduced from 50 to prevent excessive overfitting
output_dir = "outputs_llama3.3"
#output_dir = "outputsllama3.370b"



# ====== INITIALIZE EVALUATION TOOLS ======
try:
    import nltk
    nltk.download('punkt', quiet=True)
    from nltk.tokenize import word_tokenize
except ImportError:
    import nltk
    nltk.download('punkt', quiet=True)
    from nltk.tokenize import word_tokenize

try:
    from rouge import Rouge
except ImportError:
    from rouge import Rouge

# ====== CUSTOM CALLBACKS ======
# 1. Early Stopping Callback
class EarlyStoppingCallback(TensorBoardCallback):
    def __init__(self, patience=500, min_delta=0.01):  # Reduced patience from 3000 to 500
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False
        self.best_model_step = 0
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        super().on_log(args, state, control, logs, **kwargs)
        if logs and "loss" in logs:
            current_loss = logs["loss"]
            current_step = state.global_step
            
            # Check if loss improved
            if self.best_loss - current_loss > self.min_delta:
                self.best_loss = current_loss
                self.counter = 0
                self.best_model_step = current_step
                print(f"Step {current_step}: New best loss: {current_loss:.6f}")
            else:
                self.counter += 1
                print(f"Step {current_step}: No improvement for {self.counter} steps. Best: {self.best_loss:.6f}, Current: {current_loss:.6f}")
                
            # Check if we should stop
            if self.counter >= self.patience:
                self.should_stop = True
                control.should_training_stop = True
                print(f"\nEarly stopping triggered! No improvement for {self.patience} steps.")
                print(f"Best loss: {self.best_loss:.6f} at step {self.best_model_step}")

# 2. Loss Logging Callback (Enhanced)
class LossLoggingCallback(TensorBoardCallback):
    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = csv_path
        # Create CSV file with header
        with open(self.csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['epoch', 'step', 'loss', 'perplexity', 'learning_rate'])
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        super().on_log(args, state, control, logs, **kwargs)
        if logs and "loss" in logs:
            current_epoch = state.epoch
            current_step = state.global_step
            current_loss = logs["loss"]
            current_perplexity = np.exp(current_loss)  # Calculate perplexity as exp(loss)
            current_lr = logs.get("learning_rate", 0)
            
            # Store the loss in CSV
            with open(self.csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([current_epoch, current_step, current_loss, current_perplexity, current_lr])
            
            # Store in memory for plotting later
            all_losses.append({
                'epoch': current_epoch,
                'step': current_step,
                'loss': current_loss,
                'perplexity': current_perplexity,
                'learning_rate': current_lr
            })
            
            # Log to TensorBoard
            tb_writer.add_scalar('train/loss', current_loss, current_step)
            tb_writer.add_scalar('train/perplexity', current_perplexity, current_step)
            tb_writer.add_scalar('train/learning_rate', current_lr, current_step)
            
            # Print loss every 10 steps
            if current_step % 10 == 0:
                print(f"Step {current_step}: Loss = {current_loss:.4f}, Perplexity = {current_perplexity:.4f}, LR = {current_lr:.8f}")

# 3. Gradient Monitoring Callback
class GradientMonitoringCallback(TensorBoardCallback):
    def __init__(self, model, log_freq=50):
        super().__init__()
        self.model = model
        self.log_freq = log_freq
        
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_freq == 0:  # Monitor periodically
            grad_norms = []
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_norms.append(grad_norm)
                    
                    # Log to TensorBoard
                    tb_writer.add_scalar(f'gradients/{name}_norm', grad_norm, state.global_step)
                    
                    # Alert if gradients are extremely small or large
                    if grad_norm > 10.0:
                        print(f"Warning: Large gradient for {name}: {grad_norm}")
                    elif grad_norm < 1e-8 and grad_norm > 0:
                        print(f"Warning: Very small gradient for {name}: {grad_norm}")
            
            # Log global statistics
            if grad_norms:
                tb_writer.add_scalar('gradients/mean_norm', np.mean(grad_norms), state.global_step)
                tb_writer.add_scalar('gradients/max_norm', np.max(grad_norms), state.global_step)
                tb_writer.add_scalar('gradients/min_norm', np.min(grad_norms), state.global_step)

# ====== DATA LOADING ======
def load_jsonl(file_path, num_samples=None):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    if num_samples and num_samples < len(data):
        data = data[:num_samples]
    
    return data

# Function to check if text contains code (common code indicators)
def contains_code(text):
    code_indicators = [
        "```"
    ]
    
    for indicator in code_indicators:
        if indicator in text:
            return True
    return False


# Load and filter data
print(f"Loading data: Train.jsonl ({train_samples} samples)")
raw_train_data = load_jsonl("Train_with_outliers.jsonl", train_samples)

print(f"Loading data: Validation.jsonl ({validation_samples} samples)")
raw_validation_data = load_jsonl("Validation_with_outliers.jsonl", test_samples)  # Using Train.jsonl for test data as well

print(f"Loading data: Test.jsonl ({test_samples} samples)")
raw_test_data = load_jsonl("Test_with_outliers.jsonl", test_samples)  # Using Train.jsonl for test data as well

filtered_train_data = []
for idx, item in enumerate(raw_train_data, start=1):
    if isinstance(item, dict) and "prompt" in item and "completion" in item:
            filtered_train_data.append(item)

filtered_validation_data = []
for idx, item in enumerate(raw_train_data, start=1):
    if isinstance(item, dict) and "prompt" in item and "completion" in item:
            filtered_validation_data.append(item)

filtered_test_data = []
for idx, item in enumerate(raw_train_data, start=1):
    if isinstance(item, dict) and "prompt" in item and "completion" in item:
            filtered_test_data.append(item)

print(f"Filtered training items: {len(filtered_train_data)}")
print(f"Filtered validation items: {len(filtered_validation_data)}")
print(f"Filtered test items: {len(filtered_test_data)}")

# Take required number of samples
train_data = filtered_train_data[:train_samples]
val_data = filtered_validation_data[:validation_samples]
test_data = filtered_test_data[:test_samples]
 
print(f"Selected {len(train_data)} training samples without code")
print(f"Selected {len(test_data)} test samples without code")
print(f"Selected {len(val_data)} validation samples without code")

# Create conversations in the correct format for the model
def create_conversations(data):
    conversations = []
    for item in data:
        conversation = [
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["completion"]}
        ]
        conversations.append({"conversations": conversation})
    return conversations

# ====== MODEL LOADING ======
print(f"Loading model: {model_name}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Add LoRA adapters with improved settings
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,  # Increased from 8
    lora_dropout=0.1,  # Increased from previous values
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    use_rslora=True,  # Enable rank-stabilized LoRA for better optimization
    loftq_config=None,
)

# Set up chat template
tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",
)

# ====== DATASET PREPARATION ======
# Create datasets
train_dataset = Dataset.from_list(create_conversations(train_data))
val_dataset = Dataset.from_list(create_conversations(val_data))

# Prepare the datasets
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts}

train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
val_dataset = val_dataset.map(formatting_prompts_func, batched=True)

# Make sure output directory exists
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

# Create a list to store all loss values
all_losses = []

# Create a TensorBoard writer
tb_writer = SummaryWriter(os.path.join(output_dir, "logs"))

# Function to configure publication-quality plot style
def setup_publication_plot_style():
    """Configure matplotlib for publication-quality plots"""
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams['figure.dpi'] = 300
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif']
    mpl.rcParams['font.size'] = 11
    mpl.rcParams['axes.titlesize'] = 12
    mpl.rcParams['axes.labelsize'] = 11
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['ytick.labelsize'] = 10
    mpl.rcParams['legend.fontsize'] = 10
    mpl.rcParams['lines.linewidth'] = 2.0
    mpl.rcParams['axes.linewidth'] = 1.0
    mpl.rcParams['xtick.major.width'] = 1.0
    mpl.rcParams['ytick.major.width'] = 1.0
    mpl.rcParams['grid.linewidth'] = 0.8
    mpl.rcParams['grid.alpha'] = 0.3
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[
        '#0173B2', '#DE8F05', '#029E73', '#D55E00', '#CC78BC', 
        '#CA9161', '#FBAFE4', '#949494', '#ECE133', '#56B4E9'
    ])

# Function to plot the loss curve
def plot_loss_curve():
    """Create publication-quality plots of the training loss and perplexity"""
    if not all_losses:  # Check if the list is empty
        print("Warning: No loss data available to plot.")
        return
        
    setup_publication_plot_style()
    
    # Convert all_losses to DataFrame
    loss_df = pd.DataFrame(all_losses)
    
    # Check if required columns exist
    required_columns = ['step', 'loss']
    missing_columns = [col for col in required_columns if col not in loss_df.columns]
    if missing_columns:
        print(f"Warning: Missing columns in loss data: {missing_columns}")
        # If 'step' is missing but we have 'global_step', use that instead
        if 'step' in missing_columns and 'global_step' in loss_df.columns:
            loss_df['step'] = loss_df['global_step']
            missing_columns.remove('step')
        # If still missing required columns, we can't plot
        if missing_columns:
            print("Cannot generate loss plot due to missing data.")
            return
    
    # Save as CSV (in addition to the per-step CSV we're already creating)
    loss_df.to_csv(os.path.join(output_dir, "loss_summary.csv"), index=False)
    
    # Create a figure with two subplots - one for loss, one for perplexity
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)
    
    # Plot 1: Loss curve
    ax1 = plt.subplot(gs[0])
    
    ax1.plot(loss_df['step'], loss_df['loss'], 
            marker='o', markersize=4, linestyle='-', 
            color='#0173B2', alpha=0.7, markevery=5,
            markerfacecolor='white', markeredgewidth=1.5)
    
    # Add a trend line using moving average
    window_size = min(15, len(loss_df) // 5) if len(loss_df) > 20 else 1
    if window_size > 1:
        loss_df['trend'] = loss_df['loss'].rolling(window=window_size, center=True).mean()
        ax1.plot(loss_df['step'], loss_df['trend'], 
                linestyle='-', linewidth=2, color='#D55E00', alpha=0.9,
                label=f'Trend (MA{window_size})')
    
    ax1.set_title('Training Loss', fontweight='bold', pad=15)
    ax1.set_xlabel('Training Steps', fontweight='bold', labelpad=10)
    ax1.set_ylabel('Loss', fontweight='bold', labelpad=10)
    
    # Set axis limits with some padding
    ymin = max(0, loss_df['loss'].min() * 0.95)
    ymax = loss_df['loss'].max() * 1.05
    ax1.set_ylim(ymin, ymax)
    
    # Use professional gridlines
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Add annotations for initial and final loss
    initial_loss = loss_df['loss'].iloc[0]
    final_loss = loss_df['loss'].iloc[-1]
    ax1.annotate(f'Initial: {initial_loss:.4f}', 
                xy=(loss_df['step'].iloc[0], initial_loss),
                xytext=(10, 15), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    ax1.annotate(f'Final: {final_loss:.4f}', 
                xy=(loss_df['step'].iloc[-1], final_loss),
                xytext=(-10, -20), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-.2'))
    
    # Add legend if we have a trend line
    if window_size > 1:
        ax1.legend(loc='upper right')
    
    # Plot 2: Perplexity curve
    ax2 = plt.subplot(gs[1])
    
    ax2.plot(loss_df['step'], loss_df['perplexity'], 
            marker='o', markersize=4, linestyle='-', 
            color='#029E73', alpha=0.7, markevery=5,
            markerfacecolor='white', markeredgewidth=1.5)
    
    # Add a trend line for perplexity
    if window_size > 1:
        loss_df['perplexity_trend'] = loss_df['perplexity'].rolling(window=window_size, center=True).mean()
        ax2.plot(loss_df['step'], loss_df['perplexity_trend'], 
                linestyle='-', linewidth=2, color='#CC78BC', alpha=0.9,
                label=f'Trend (MA{window_size})')
    
    ax2.set_title('Training Perplexity', fontweight='bold', pad=15)
    ax2.set_xlabel('Training Steps', fontweight='bold', labelpad=10)
    ax2.set_ylabel('Perplexity (exp(loss))', fontweight='bold', labelpad=10)
    
    # Set axis limits with some padding for perplexity
    ymin_ppl = max(1.0, loss_df['perplexity'].min() * 0.95)  # Perplexity should be >= 1
    ymax_ppl = loss_df['perplexity'].max() * 1.05
    ax2.set_ylim(ymin_ppl, ymax_ppl)
    
    # Use professional gridlines
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add annotations for initial and final perplexity
    initial_ppl = loss_df['perplexity'].iloc[0]
    final_ppl = loss_df['perplexity'].iloc[-1]
    ax2.annotate(f'Initial: {initial_ppl:.4f}', 
                xy=(loss_df['step'].iloc[0], initial_ppl),
                xytext=(10, 15), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    ax2.annotate(f'Final: {final_ppl:.4f}', 
                xy=(loss_df['step'].iloc[-1], final_ppl),
                xytext=(-10, -20), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-.2'))
    
    # Add legend if we have a trend line
    if window_size > 1:
        ax2.legend(loc='upper right')
    
    # Add descriptive text
    plt.figtext(0.02, 0.02, f"Total steps: {len(loss_df)}", 
                ha="left", fontsize=8, fontstyle='italic')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "loss_and_perplexity.png"), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "plots", "loss_and_perplexity.pdf"), bbox_inches='tight')
    plt.close()
    
    # Additional plot just for perplexity
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    ax.plot(loss_df['step'], loss_df['perplexity'], 
            marker='o', markersize=4, linestyle='-', 
            color='#029E73', alpha=0.7, markevery=5,
            markerfacecolor='white', markeredgewidth=1.5)
    
    if window_size > 1:
        ax.plot(loss_df['step'], loss_df['perplexity_trend'], 
                linestyle='-', linewidth=2, color='#CC78BC', alpha=0.9,
                label=f'Trend (MA{window_size})')
    
    ax.set_title('Training Perplexity', fontweight='bold', pad=15)
    ax.set_xlabel('Training Steps', fontweight='bold', labelpad=10)
    ax.set_ylabel('Perplexity', fontweight='bold', labelpad=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    if window_size > 1:
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "perplexity_curve.png"), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "plots", "perplexity_curve.pdf"), bbox_inches='tight')
    plt.close()
    
    # Plot average perplexity per epoch if epoch data is available
    if 'epoch' in loss_df.columns and len(loss_df['epoch'].unique()) > 1:
        # Compute average perplexity per epoch
        avg_perplexity_by_epoch = loss_df.groupby('epoch')['perplexity'].mean().reset_index()
        std_perplexity_by_epoch = loss_df.groupby('epoch')['perplexity'].std().reset_index()
        
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Plot with error bars
        ax.errorbar(avg_perplexity_by_epoch['epoch'], avg_perplexity_by_epoch['perplexity'],
                    yerr=std_perplexity_by_epoch['perplexity'], 
                    marker='o', markersize=8, linestyle='-', linewidth=2,
                    color='#029E73', ecolor='#CC78BC', capsize=5, 
                    markerfacecolor='white', markeredgewidth=2)
        
        # Add a trend line
        z = np.polyfit(avg_perplexity_by_epoch['epoch'], avg_perplexity_by_epoch['perplexity'], 1)
        p = np.poly1d(z)
        ax.plot(avg_perplexity_by_epoch['epoch'], p(avg_perplexity_by_epoch['epoch']), 
                linestyle='--', linewidth=1.5, color='#D55E00', alpha=0.8,
                label=f'Trend: {z[0]:.4f}x + {z[1]:.4f}')
        
        ax.set_title('Average Perplexity per Epoch', fontweight='bold', pad=15)
        ax.set_xlabel('Epoch', fontweight='bold', labelpad=10)
        ax.set_ylabel('Average Perplexity', fontweight='bold', labelpad=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Force x-axis to use integer ticks for epochs
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add annotations for each point
        for i, (epoch, perplexity) in enumerate(zip(avg_perplexity_by_epoch['epoch'], avg_perplexity_by_epoch['perplexity'])):
            ax.annotate(f'{perplexity:.4f}', 
                        xy=(epoch, perplexity),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
        
        ax.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plots", "avg_perplexity_by_epoch.png"), 
                    bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, "plots", "avg_perplexity_by_epoch.pdf"), 
                    bbox_inches='tight')
        plt.close()

# ====== EVALUATION FUNCTION ======
def evaluate_model(model, tokenizer, test_data, epoch):
    """Evaluate model on test data and compute metrics"""
    print(f"\n===== Evaluating model after Epoch {epoch} =====")
    
    # Enable inference mode
    FastLanguageModel.for_inference(model)
    
    # Setup metrics
    rouge = Rouge()
    smooth = SmoothingFunction().method1
    
    predictions = []
    references = []
    prompts = []
    bleu_scores = []
    rouge_scores = []
    results_data = []
    total_tokens = 0
    total_log_likelihood = 0
    
    # Generate predictions for test data
    for i, item in enumerate(test_data):
        prompt = item["prompt"]
        reference = item["completion"]
        prompts.append(prompt)
        
        # Prepare input
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Generate prediction
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=1024,
                use_cache=True,
                temperature=0.7,
                min_p=0.1,
                output_scores=True,
                return_dict_in_generate=True
            )
        
        # Get token scores for calculating perplexity
        token_scores = outputs.scores
        token_ids = outputs.sequences[0][inputs.size(1):] # Only generated tokens
        
        # Calculate log likelihood
        log_likelihood = 0
        for token_idx, token_id in enumerate(token_ids):
            if token_idx < len(token_scores):  # Ensure we have scores for this token
                token_score = token_scores[token_idx]
                log_prob = torch.log_softmax(token_score, dim=-1)[0, token_id]
                log_likelihood += log_prob.item()
                
        # Count tokens
        num_tokens = len(token_ids)
        total_tokens += num_tokens
        total_log_likelihood += log_likelihood
                
        # Decode prediction
        full_output = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        prediction = full_output.split("assistant\n\n")[1]
        
        predictions.append(prediction)
        references.append(reference)
        
        # Calculate BLEU
        ref_tokens = word_tokenize(reference.lower())
        pred_tokens = word_tokenize(prediction.lower())
        
        # BLEU scores (1, 2, 3, 4-gram)
        bleu1 = sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth)
        bleu2 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
        bleu3 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth)
        bleu4 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
        
        bleu_scores.append({
            'bleu-1': bleu1,
            'bleu-2': bleu2,
            'bleu-3': bleu3,
            'bleu-4': bleu4
        })
        
        # Calculate ROUGE
        try:
            rouge_score = rouge.get_scores(prediction, reference)[0]
            rouge_scores.append(rouge_score)
        except Exception as e:
            print(f"Rouge error on sample {i}: {e}")
            rouge_scores.append({'rouge-1': {'f': 0}, 'rouge-2': {'f': 0}, 'rouge-l': {'f': 0}})
        
        # Store individual result data
        results_data.append({
            'sample_id': i,
            'prompt': prompt,
            'actual': reference,
            'predicted': prediction,
            'bleu-1': bleu1,
            'bleu-2': bleu2,
            'bleu-3': bleu3,
            'bleu-4': bleu4,
            'rouge-1': rouge_scores[-1]['rouge-1']['f'],
            'rouge-2': rouge_scores[-1]['rouge-2']['f'],
            'rouge-l': rouge_scores[-1]['rouge-l']['f']
        })
            
        print(f"Sample {i+1}/{len(test_data)} processed")
    
    # Calculate perplexity
    avg_neg_log_likelihood = -total_log_likelihood / total_tokens if total_tokens > 0 else 0
    perplexity = np.exp(avg_neg_log_likelihood)
    
    # Aggregate metrics
    avg_bleu1 = np.mean([score['bleu-1'] for score in bleu_scores])
    avg_bleu2 = np.mean([score['bleu-2'] for score in bleu_scores])
    avg_bleu3 = np.mean([score['bleu-3'] for score in bleu_scores])
    avg_bleu4 = np.mean([score['bleu-4'] for score in bleu_scores])
    
    avg_rouge1 = np.mean([score['rouge-1']['f'] for score in rouge_scores])
    avg_rouge2 = np.mean([score['rouge-2']['f'] for score in rouge_scores])
    avg_rougeL = np.mean([score['rouge-l']['f'] for score in rouge_scores])
    
    # Print metrics
    print(f"\nEvaluation Metrics - Epoch {epoch}:")
    print(f"Perplexity: {perplexity:.4f}")
    print(f"BLEU-1: {avg_bleu1:.4f}")
    print(f"BLEU-2: {avg_bleu2:.4f}")
    print(f"BLEU-3: {avg_bleu3:.4f}")
    print(f"BLEU-4: {avg_bleu4:.4f}")
    print(f"ROUGE-1: {avg_rouge1:.4f}")
    print(f"ROUGE-2: {avg_rouge2:.4f}")
    print(f"ROUGE-L: {avg_rougeL:.4f}")
    
    # Save metrics to file
    metrics = {
        'epoch': epoch,
        'perplexity': float(perplexity),
        'bleu-1': float(avg_bleu1),
        'bleu-2': float(avg_bleu2),
        'bleu-3': float(avg_bleu3),
        'bleu-4': float(avg_bleu4),
        'rouge-1': float(avg_rouge1),
        'rouge-2': float(avg_rouge2),
        'rouge-l': float(avg_rougeL)
    }
    
    with open(os.path.join(output_dir, f"metrics_epoch_{epoch}.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save detailed results to CSV
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(os.path.join(output_dir, f"results_epoch_{epoch}.csv"), index=False)
    
    # Save example generations in a more readable format
    with open(os.path.join(output_dir, f"generations_epoch_{epoch}.txt"), 'w') as f:
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            f.write(f"Example {i+1}:\n")
            f.write(f"Prompt: {prompts[i]}\n\n")
            f.write(f"REFERENCE:\n{ref}\n\n")
            f.write(f"PREDICTION:\n{pred}\n\n")
            f.write("-" * 80 + "\n\n")
    
    # Return to training mode
    model.train()
    
    return metrics

# ====== TRAINING SETUP ======
# Using improved training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=8,  # Increased from 1 to 4
    per_device_eval_batch_size=8,   # Increased from 1 to 4
    gradient_accumulation_steps=4,  # Reduced from 8 to 2 (maintains effective batch size of 8)
    warmup_ratio=0.5,  # 10% of steps will be used for warmup
    num_train_epochs=1,  # We'll do one epoch at a time for evaluation
    learning_rate=1e-6,  # Reduced from 5e-6 to improve learning while maintaining stability
    fp16=not is_bfloat16_supported(),  # Enable mixed precision training based on hardware support
    bf16=is_bfloat16_supported(),      # Use bfloat16 if available (better numerical stability than fp16)
    logging_steps=5,  # Log training metrics every 5 steps
  
    evaluation_strategy="steps",  # Evaluate periodically during training
    eval_steps=125,  # Evaluate approximately 10 times per epoch (1250 steps / 10)
    save_strategy="steps",  # Save checkpoints based on steps
    save_steps=125,  # Save approximately 10 times per epoch
    save_total_limit=3,  # Keep only the 3 best checkpoints to save disk space
    optim="adamw_8bit",  # Use 8-bit AdamW optimizer to save memory
    weight_decay=0.05,  # Increased from 0.05 for stronger regularization
    lr_scheduler_type="cosine",  # Use cosine scheduler for better convergence
    seed=42,  # Random seed for reproducibility
    output_dir=output_dir,
    report_to="none",  # Disable default tracking systems
    max_grad_norm=1.0,  # Increase gradient clipping threshold to allow larger updates
)

# Initialize callbacks
loss_file_path = os.path.join(output_dir, "training_loss.csv")
loss_callback = LossLoggingCallback(loss_file_path)
early_stopping = EarlyStoppingCallback(patience=3000, min_delta=0.01)  # Reduced patience, increased min_delta
gradient_monitor = GradientMonitoringCallback(model, log_freq=50)  # Adjusted frequency

# Set up trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    dataset_num_proc=1,
    packing=False,
    args=training_args,
    callbacks=[loss_callback, early_stopping, gradient_monitor]
)

# Only train on assistant responses, not user prompts
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
)

# Empty CUDA cache before training
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("Cleared CUDA cache")

# Show memory stats before training
if torch.cuda.is_available():
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

all_metrics = []

# Train for specified number of epochs with evaluation after each
for epoch in range(1, num_epochs + 1):
    print(f"\n===== Starting Epoch {epoch}/{num_epochs} =====")
    
    # Train for one epoch
    trainer_stats = trainer.train()
    
    # Save checkpoint after each epoch
    epoch_output_dir = os.path.join(output_dir, f"epoch_{epoch}")
    os.makedirs(epoch_output_dir, exist_ok=True)
    model.save_pretrained(epoch_output_dir)
    tokenizer.save_pretrained(epoch_output_dir)
    print(f"Saved model checkpoint for epoch {epoch}")
    
    # Clear CUDA cache before evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Evaluate on test data
    metrics = evaluate_model(model, tokenizer, test_data, epoch)
    all_metrics.append(metrics)
    
    # Check if early stopping was triggered
    if early_stopping.should_stop:
        print(f"Early stopping triggered - halting training after epoch {epoch}")
        break
    
    # Re-initialize trainer for next epoch with cosine-decayed learning rate
    if epoch < num_epochs:
        # Calculate new learning rate with cosine decay
        progress = epoch / num_epochs
        new_lr = 1e-5 * (0.5 * (1 + np.cos(np.pi * progress)))
        
        # Create new training arguments with updated learning rate
        next_epoch_args = TrainingArguments(
            per_device_train_batch_size=4,  # Increased batch size
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,
            warmup_steps=50,  # Increased from 5
            num_train_epochs=1,
            learning_rate=new_lr,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=5,
            evaluation_strategy="steps",
            eval_steps=125,  # Adjusted for 10,000 sample dataset
            save_strategy="steps",
            save_steps=125,  # Adjusted for 10,000 sample dataset
            save_total_limit=3,
            optim="adamw_8bit",
            weight_decay=0.1,  # Increased from 0.05
            lr_scheduler_type="cosine",
            seed=3407,  # Different seed for each epoch
            output_dir=os.path.join(output_dir, f"epoch_{epoch+1}"),
            report_to="none",
            max_grad_norm=2.0,  # Increased from 1.0
        )
        
        # Reinitialize the trainer with updated arguments
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
            dataset_num_proc=1,
            packing=False,
            args=next_epoch_args,
            callbacks=[loss_callback, early_stopping, gradient_monitor]
        )
        
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
            response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
        )

# Show final memory and time stats
if torch.cuda.is_available():
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# ====== SAVE FINAL MODEL ======
print("Saving final model to 'finetuned_model'")
model.save_pretrained("finetuned_model")
tokenizer.save_pretrained("finetuned_model")

# Generate final test predictions using the best model
print("Generating final predictions with the trained model...")

# Enable inference mode
FastLanguageModel.for_inference(model)

final_predictions = []
for i, item in enumerate(test_data):
    prompt = item["prompt"]
    
    # Prepare input
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate prediction
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=1024,
            use_cache=True,
            temperature=0.7,
            min_p=0.1
        )
    
    # Decode prediction
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prediction = full_output.split("assistant")[1]
    
    final_predictions.append(prediction)
    print(f"Final prediction {i+1}/{len(test_data)} generated")

# Save test data comparison
def save_test_data_comparison(test_data, predictions, output_path):
    """Save test data and predictions in multiple formats for easy comparison"""
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Prepare data
    formatted_data = []
    for i, (test_item, pred) in enumerate(zip(test_data, predictions)):
        formatted_data.append({
            'id': i,
            'prompt': test_item['prompt'],
            'actual': test_item['completion'],
            'generated': pred
        })
    
    # Save as CSV
    df = pd.DataFrame(formatted_data)
    df.to_csv(f"{output_path}.csv", index=False)
    
    # Save as JSON
    with open(f"{output_path}.json", 'w') as f:
        json.dump(formatted_data, f, indent=2)
    
    # Save as text
    with open(f"{output_path}.txt", 'w') as f:
        for item in formatted_data:
            f.write(f"ID: {item['id']}\n")
            f.write(f"PROMPT:\n{item['prompt']}\n\n")
            f.write(f"ACTUAL:\n{item['actual']}\n\n")
            f.write(f"GENERATED:\n{item['generated']}\n\n")
            f.write("-" * 80 + "\n\n")
    
    # Create a markdown table version
    with open(f"{output_path}.md", 'w') as f:
        f.write("# Test Data Comparison\n\n")
        
        for item in formatted_data:
            f.write(f"## Sample {item['id']}\n\n")
            f.write(f"### Prompt\n\n{item['prompt']}\n\n")
            f.write(f"### Actual\n\n{item['actual']}\n\n")
            f.write(f"### Generated\n\n{item['generated']}\n\n")
            f.write("---\n\n")
    
    return formatted_data

# Save the final test data comparison
save_test_data_comparison(
    test_data, 
    final_predictions, 
    os.path.join(output_dir, "final_test_comparison")
)

# Save all metrics in one file
with open(os.path.join(output_dir, "all_metrics.json"), 'w') as f:
    json.dump(all_metrics, f, indent=2)

# Run final plotting functions
plot_loss_curve()

print("\nTraining and evaluation complete!")
print("Final model saved to 'finetuned_model'")
print(f"Metrics and plots saved to '{output_dir}'")
print(f"Test data comparisons saved to '{output_dir}/final_test_comparison.[csv|json|txt|md]'")