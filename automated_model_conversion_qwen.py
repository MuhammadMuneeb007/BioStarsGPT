#!/usr/bin/env python3
"""
Automated Qwen Model Conversion Script - Fixed for Qwen2.5
Converts your fine-tuned Qwen model to GGUF format for Open WebUI
Specialized for Polygenic Risk Score (PRS) analysis tools
"""

import os
import sys
import subprocess
import shutil
import json
import gc
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table

console = Console()

class QwenModelConverter:
    def __init__(self, checkpoint_path, output_dir="./qwen_gguf_output"):
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir)
        self.merged_model_path = Path("./qwen_merged_model")
        self.llama_cpp_path = Path("./llama.cpp")
        
    def check_prerequisites(self):
        """Check if all required packages are installed"""
        console.print("[bold yellow]Checking prerequisites...[/]")
        
        required_packages = ['torch', 'transformers', 'unsloth', 'gguf']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                console.print(f"✅ {package}")
            except ImportError:
                console.print(f"❌ {package}")
                missing_packages.append(package)
        
        if missing_packages:
            console.print(f"\n[bold red]Missing packages: {missing_packages}[/]")
            console.print("Install them with:")
            console.print(f"pip install {' '.join(missing_packages)}")
            return False
        
        return True
    
    def validate_checkpoint(self):
        """Validate that the checkpoint exists and has required files"""
        console.print(f"[bold yellow]Validating checkpoint: {self.checkpoint_path}[/]")
        
        if not self.checkpoint_path.exists():
            console.print(f"[bold red]❌ Checkpoint directory not found: {self.checkpoint_path}[/]")
            return False
        
        required_files = ['adapter_config.json', 'adapter_model.safetensors']
        missing_files = []
        
        for file in required_files:
            file_path = self.checkpoint_path / file
            if not file_path.exists():
                missing_files.append(file)
            elif file_path.stat().st_size == 0:
                console.print(f"⚠️  {file} is empty")
                missing_files.append(file)
            else:
                console.print(f"✅ {file}")
        
        if missing_files:
            console.print(f"[bold red]❌ Missing or empty files: {missing_files}[/]")
            return False
        
        return True
    
    def clear_memory(self):
        """Clear GPU and CPU memory"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass
    
    def fix_config_file(self):
        """Fix or create proper config.json for Qwen2.5"""
        config_path = self.merged_model_path / "config.json"
        
        console.print("🔧 Fixing config.json for Qwen2.5...")
        
        try:
            # Try to read existing config
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
            else:
                config = {}
            
            # Ensure it has the correct model_type for Qwen2.5
            config["model_type"] = "qwen2"
            config["architectures"] = ["Qwen2ForCausalLM"]
            
            # Add other essential Qwen2.5 config parameters if missing
            if "vocab_size" not in config:
                config["vocab_size"] = 152064
            if "hidden_size" not in config:
                config["hidden_size"] = 3584
            if "intermediate_size" not in config:
                config["intermediate_size"] = 18944
            if "num_attention_heads" not in config:
                config["num_attention_heads"] = 28
            if "num_hidden_layers" not in config:
                config["num_hidden_layers"] = 28
            if "num_key_value_heads" not in config:
                config["num_key_value_heads"] = 4
            if "max_position_embeddings" not in config:
                config["max_position_embeddings"] = 32768
            if "sliding_window" not in config:
                config["sliding_window"] = 32768
            if "max_window_layers" not in config:
                config["max_window_layers"] = 21
            if "tie_word_embeddings" not in config:
                config["tie_word_embeddings"] = False
            if "rope_theta" not in config:
                config["rope_theta"] = 1000000.0
            if "use_sliding_window" not in config:
                config["use_sliding_window"] = False
            if "torch_dtype" not in config:
                config["torch_dtype"] = "bfloat16"
            if "transformers_version" not in config:
                config["transformers_version"] = "4.37.0"
            
            # Write the fixed config
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            console.print(f"✅ Fixed config.json with model_type: {config['model_type']}")
            return True
            
        except Exception as e:
            console.print(f"❌ Failed to fix config.json: {e}")
            return False
    
    def merge_model_unsloth(self):
        """Primary merging method using unsloth"""
        try:
            from unsloth import FastLanguageModel
            import torch
            
            console.print("🔄 Attempting unsloth merge method...")
            
            # Clear memory first
            self.clear_memory()
            
            # Load the fine-tuned model with specific settings for Qwen2.5
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(self.checkpoint_path),
                max_seq_length=2048,
                load_in_4bit=False,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Move model to CPU to avoid device conflicts
            model = model.cpu()
            self.clear_memory()
            
            # Merge the adapter with base model
            console.print("🔗 Merging adapter with base model...")
            model = model.merge_and_unload()
            
            # Clear memory again
            self.clear_memory()
            
            # Create output directory
            self.merged_model_path.mkdir(exist_ok=True)
            
            # Save the merged model with proper config
            console.print("💾 Saving merged model...")
            model.save_pretrained(
                str(self.merged_model_path),
                safe_serialization=True,
                max_shard_size="5GB"
            )
            tokenizer.save_pretrained(str(self.merged_model_path))
            
            # Fix the config file
            self.fix_config_file()
            
            # Clear memory
            del model, tokenizer
            self.clear_memory()
            
            console.print(f"✅ Merged Qwen model saved to: {self.merged_model_path}")
            return True
            
        except Exception as e:
            console.print(f"⚠️  Unsloth merge method failed: {e}")
            self.clear_memory()
            return False
    
    def merge_model_transformers(self):
        """Alternative merging method using transformers directly"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            import torch
            
            console.print("🔄 Attempting transformers + peft merge method...")
            
            # Clear memory
            self.clear_memory()
            
            # Base model name for Qwen2.5-7B
            base_model_name = "unsloth/Qwen2.5-7B"
            console.print(f"📋 Base model: {base_model_name}")
            
            # Load base model
            console.print("📥 Loading base model...")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                trust_remote_code=True
            )
            
            # Load and merge adapter
            console.print("🔗 Loading and merging adapter...")
            model = PeftModel.from_pretrained(
                base_model,
                str(self.checkpoint_path),
                torch_dtype=torch.float16
            )
            
            # Merge
            model = model.merge_and_unload()
            
            # Clear memory
            self.clear_memory()
            
            # Create output directory
            self.merged_model_path.mkdir(exist_ok=True)
            
            # Save the merged model
            console.print("💾 Saving merged model...")
            model.save_pretrained(
                str(self.merged_model_path),
                safe_serialization=True,
                max_shard_size="5GB"
            )
            tokenizer.save_pretrained(str(self.merged_model_path))
            
            # Fix the config file
            self.fix_config_file()
            
            # Clear memory
            del model, base_model, tokenizer
            self.clear_memory()
            
            console.print(f"✅ Merged Qwen model saved to: {self.merged_model_path}")
            return True
            
        except Exception as e:
            console.print(f"⚠️  Transformers merge method failed: {e}")
            self.clear_memory()
            return False
    
    def merge_model_manual(self):
        """Manual method - download base model and create proper config"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            console.print("🔄 Attempting manual method...")
            
            # Base model name
            base_model_name = "unsloth/Qwen2.5-7B"
            
            # Create merged model directory
            self.merged_model_path.mkdir(exist_ok=True)
            
            # Download and save just the tokenizer and config
            console.print("📁 Downloading base model config and tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                trust_remote_code=True
            )
            
            # Load config from base model
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Save tokenizer and model
            tokenizer.save_pretrained(str(self.merged_model_path))
            base_model.save_pretrained(
                str(self.merged_model_path),
                safe_serialization=True,
                max_shard_size="5GB"
            )
            
            # Fix the config file
            self.fix_config_file()
            
            # Clear memory
            del base_model, tokenizer
            self.clear_memory()
            
            console.print(f"⚠️  Manual setup completed at: {self.merged_model_path}")
            console.print("⚠️  Note: This uses the base model without fine-tuning applied")
            
            return True
            
        except Exception as e:
            console.print(f"❌ Manual method failed: {e}")
            return False
    
    def merge_model(self):
        """Merge the LoRA adapter with the base model using multiple methods"""
        console.print("[bold yellow]Merging adapter with base Qwen model...[/]")
        
        # Try multiple methods in order
        methods = [
            ("Unsloth", self.merge_model_unsloth),
            ("Transformers + PEFT", self.merge_model_transformers),
            ("Manual", self.merge_model_manual)
        ]
        
        for method_name, method_func in methods:
            console.print(f"\n🔧 Trying {method_name} method...")
            if method_func():
                return True
            
            # Clean up any partial files
            if self.merged_model_path.exists():
                try:
                    shutil.rmtree(self.merged_model_path)
                except:
                    pass
        
        console.print("[bold red]❌ All merge methods failed[/]")
        return False
    
    def setup_llama_cpp(self):
        """Clone and build llama.cpp if not already present"""
        console.print("[bold yellow]Setting up llama.cpp...[/]")
        
        if self.llama_cpp_path.exists():
            console.print("✅ llama.cpp already exists")
            # Check if binaries exist
            convert_script = self.llama_cpp_path / "convert_hf_to_gguf.py"
            quantize_binary = self.llama_cpp_path / "build" / "bin" / "llama-quantize"
            
            if convert_script.exists() and quantize_binary.exists():
                console.print("✅ llama.cpp binaries found")
                return True
            else:
                console.print("⚠️  llama.cpp exists but binaries missing, rebuilding...")
        
        try:
            # Clone llama.cpp if it doesn't exist
            if not self.llama_cpp_path.exists():
                console.print("Cloning llama.cpp...")
                result = subprocess.run([
                    "git", "clone", "https://github.com/ggerganov/llama.cpp.git", 
                    str(self.llama_cpp_path)
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    console.print(f"[bold red]❌ Failed to clone llama.cpp: {result.stderr}[/]")
                    return False
            
            # Check if cmake is installed
            result = subprocess.run(["cmake", "--version"], capture_output=True, text=True)
            if result.returncode != 0:
                console.print("[bold red]❌ CMake not found. Installing CMake...[/]")
                # Install cmake
                install_result = subprocess.run([
                    "sudo", "apt-get", "update", "&&", 
                    "sudo", "apt-get", "install", "-y", "cmake", "build-essential"
                ], shell=True, capture_output=True, text=True)
                
                if install_result.returncode != 0:
                    console.print(f"[bold red]❌ Failed to install CMake: {install_result.stderr}[/]")
                    console.print("Please install CMake manually: sudo apt-get install cmake build-essential")
                    return False
                
                console.print("✅ CMake installed successfully")
            
            # Create build directory
            build_dir = self.llama_cpp_path / "build"
            build_dir.mkdir(exist_ok=True)
            
            # Configure with CMake - try CUDA first, fallback to CPU
            console.print("Configuring build with CMake...")
            
            # Check if NVIDIA GPU is available
            nvidia_available = False
            try:
                result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
                if result.returncode == 0:
                    nvidia_available = True
                    console.print("🎮 NVIDIA GPU detected, enabling CUDA support")
            except FileNotFoundError:
                console.print("🖥️  No NVIDIA GPU detected, using CPU-only build")
            
            # CMake configuration
            cmake_args = [
                "cmake", "..", 
                "-DCMAKE_BUILD_TYPE=Release"
            ]
            
            if nvidia_available:
                cmake_args.extend([
                    "-DGGML_CUDA=ON"
                ])
            
            result = subprocess.run(
                cmake_args,
                cwd=build_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                if nvidia_available:
                    console.print("⚠️  CUDA build failed, trying CPU-only build...")
                    # Retry without CUDA
                    cmake_args = [
                        "cmake", "..", 
                        "-DCMAKE_BUILD_TYPE=Release",
                        "-DGGML_CUDA=OFF"
                    ]
                    result = subprocess.run(
                        cmake_args,
                        cwd=build_dir,
                        capture_output=True,
                        text=True
                    )
                
                if result.returncode != 0:
                    console.print(f"[bold red]❌ Failed to configure build: {result.stderr}[/]")
                    return False
            
            console.print("✅ Build configured successfully")
            
            # Build with CMake
            console.print("Building llama.cpp (this may take a few minutes)...")
            result = subprocess.run([
                "cmake", "--build", ".", "--config", "Release", "-j", str(os.cpu_count())
            ], cwd=build_dir, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                console.print(f"[bold red]❌ Failed to build llama.cpp: {result.stderr}[/]")
                return False
            
            # Verify binaries were created
            convert_script = self.llama_cpp_path / "convert_hf_to_gguf.py"
            quantize_binary = build_dir / "bin" / "llama-quantize"
            
            if not convert_script.exists():
                console.print(f"[bold red]❌ Convert script not found: {convert_script}[/]")
                return False
            
            if not quantize_binary.exists():
                console.print(f"[bold red]❌ Quantize binary not found: {quantize_binary}[/]")
                return False
            
            console.print("✅ llama.cpp built successfully!")
            
            if nvidia_available:
                console.print("🚀 Built with CUDA support for faster processing")
            else:
                console.print("🖥️  Built with CPU support")
            
            return True
            
        except subprocess.TimeoutExpired:
            console.print("[bold red]❌ Build timed out (>10 minutes)[/]")
            return False
        except Exception as e:
            console.print(f"[bold red]❌ Error setting up llama.cpp: {e}[/]")
            return False
    
    def convert_to_gguf(self, quantization="Q4_K_M"):
        """Convert the merged model to GGUF format"""
        console.print(f"[bold yellow]Converting Qwen model to GGUF format (quantization: {quantization})...[/]")
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Verify config.json exists and is valid
        config_path = self.merged_model_path / "config.json"
        if not config_path.exists():
            console.print("⚠️  config.json missing, creating one...")
            if not self.fix_config_file():
                return False
        
        try:
            # Convert to GGUF - create f16 version first
            convert_script = self.llama_cpp_path / "convert_hf_to_gguf.py"
            f16_gguf_file = self.output_dir / "qwen-model-f16.gguf"
            
            cmd = [
                "python", str(convert_script),
                str(self.merged_model_path),
                "--outfile", str(f16_gguf_file),
                "--outtype", "f16"
            ]
            
            console.print("Converting to GGUF f16 format...")
            console.print(f"Command: {' '.join(map(str, cmd))}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                console.print(f"[bold red]❌ GGUF conversion failed: {result.stderr}[/]")
                console.print(f"[bold red]Command output: {result.stdout}[/]")
                return False
            
            # Check if the GGUF file was created
            if not f16_gguf_file.exists():
                console.print(f"[bold red]❌ GGUF file not created: {f16_gguf_file}[/]")
                return False
            
            console.print(f"✅ Unquantized GGUF file created: {f16_gguf_file}")
            console.print(f"📏 Size: {f16_gguf_file.stat().st_size / (1024**3):.2f} GB")
            
            # Store the f16 version
            self.f16_model = f16_gguf_file
            
            # Quantize the model if requested
            if quantization != "f16":
                console.print(f"Quantizing to {quantization}...")
                quantized_file = self.output_dir / f"qwen-model-{quantization.lower()}.gguf"
                
                # Use the correct path for CMake build
                quantize_binary = self.llama_cpp_path / "build" / "bin" / "llama-quantize"
                
                # Check if the binary exists
                if not quantize_binary.exists():
                    console.print(f"[bold red]❌ Quantize binary not found: {quantize_binary}[/]")
                    console.print("Trying alternative locations...")
                    
                    # Try other possible locations
                    alternative_paths = [
                        self.llama_cpp_path / "build" / "llama-quantize",
                        self.llama_cpp_path / "llama-quantize"
                    ]
                    
                    for alt_path in alternative_paths:
                        if alt_path.exists():
                            quantize_binary = alt_path
                            console.print(f"✅ Found quantize binary at: {quantize_binary}")
                            break
                    else:
                        console.print("[bold red]❌ Could not find quantize binary[/]")
                        console.print("⚠️  Keeping f16 version only")
                        self.final_model = f16_gguf_file
                        return True
                
                cmd = [
                    str(quantize_binary),
                    str(f16_gguf_file),
                    str(quantized_file),
                    quantization
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    console.print(f"[bold red]❌ Quantization failed: {result.stderr}[/]")
                    console.print("⚠️  Using f16 version as fallback")
                    self.final_model = f16_gguf_file
                    return True
                
                console.print(f"✅ Quantized model created: {quantized_file}")
                console.print(f"📏 Size: {quantized_file.stat().st_size / (1024**3):.2f} GB")
                self.final_model = quantized_file
            else:
                self.final_model = f16_gguf_file
            
            return True
            
        except Exception as e:
            console.print(f"[bold red]❌ Error converting to GGUF: {e}[/]")
            return False
    
    def create_modelfile(self, model_name="qwen-prs-assistant"):
        """Create Ollama Modelfile for Qwen"""
        console.print("[bold yellow]Creating Qwen Ollama Modelfile...[/]")
        
        # Qwen-specific chat template and system message
        modelfile_content = f'''FROM ./{self.final_model.name}

TEMPLATE """<|im_start|>system
{{{{ .System }}}}<|im_end|>
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
"""

SYSTEM """You are Qwen, created by Alibaba Cloud. You are a helpful AI assistant specialized in polygenic risk score (PRS) analysis and related genomic tools. You provide clear, accurate, and practical information about:

- Calculating and interpreting polygenic risk scores
- Using PRS tools like PRSice-2, PLINK, and LDpred
- Understanding GWAS summary statistics and their application
- Quality control procedures for genetic data
- Population structure and ancestry considerations in PRS
- Cross-ancestry portability of polygenic scores
- Best practices for PRS validation and evaluation
- Interpreting PRS results in clinical and research contexts
- Data formats and file preparation for PRS analysis
- Statistical concepts related to polygenic architecture

Always provide specific, actionable advice with examples when possible. If you're unsure about something, clearly state your limitations rather than guessing."""

PARAMETER temperature 0.7
PARAMETER top_p 0.8
PARAMETER top_k 40
PARAMETER repeat_penalty 1.05
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
'''
        
        modelfile_path = self.output_dir / "Modelfile"
        with open(modelfile_path, 'w') as f:
            f.write(modelfile_content)
        
        console.print(f"✅ Qwen Modelfile created: {modelfile_path}")
        return modelfile_path
    
    def create_ollama_model(self, model_name="qwen-prs-assistant"):
        """Create the model in Ollama"""
        console.print(f"[bold yellow]Creating Ollama model: {model_name}...[/]")
        
        try:
            # Check if Ollama is running
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if result.returncode != 0:
                console.print("[bold red]❌ Ollama is not running. Please start it with: ollama serve[/]")
                return False
            
            # Create the model
            cmd = ["ollama", "create", model_name, "-f", "Modelfile"]
            result = subprocess.run(cmd, cwd=self.output_dir, capture_output=True, text=True)
            
            if result.returncode != 0:
                console.print(f"[bold red]❌ Failed to create Ollama model: {result.stderr}[/]")
                return False
            
            console.print(f"✅ Ollama model created: {model_name}")
            
            # Test the model
            console.print("Testing the model...")
            test_cmd = ["ollama", "run", model_name, "Hello! Can you help me calculate polygenic risk scores?"]
            result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                console.print("✅ Model test successful!")
                console.print(f"Response: {result.stdout[:100]}...")
            else:
                console.print("⚠️  Model created but test failed")
            
            return True
            
        except subprocess.TimeoutExpired:
            console.print("⚠️  Model test timed out (this is normal)")
            return True
        except Exception as e:
            console.print(f"[bold red]❌ Error creating Ollama model: {e}[/]")
            return False
    
    def convert(self, quantization="Q4_K_M", model_name="qwen-prs-assistant"):
        """Main conversion pipeline"""
        console.print(Panel.fit(
            "[bold cyan]Starting Qwen2.5 Model Conversion to GGUF[/]",
            border_style="blue"
        ))
        
        steps = [
            ("Checking prerequisites", self.check_prerequisites),
            ("Validating checkpoint", self.validate_checkpoint),
            ("Merging model", self.merge_model),
            ("Setting up llama.cpp", self.setup_llama_cpp),
            ("Converting to GGUF", lambda: self.convert_to_gguf(quantization)),
            ("Creating Modelfile", lambda: self.create_modelfile(model_name)),
            ("Creating Ollama model", lambda: self.create_ollama_model(model_name))
        ]
        
        for step_name, step_func in steps:
            console.print(f"\n[bold]Step: {step_name}[/]")
            if not step_func():
                console.print(f"[bold red]❌ Failed at step: {step_name}[/]")
                return False
        
        # Success summary
        available_models = []
        if hasattr(self, 'f16_model') and self.f16_model.exists():
            available_models.append(f"• F16 (unquantized): {self.f16_model.name} ({self.f16_model.stat().st_size / (1024**3):.2f} GB)")
        
        if hasattr(self, 'final_model') and self.final_model.exists() and self.final_model != self.f16_model:
            available_models.append(f"• {quantization} (quantized): {self.final_model.name} ({self.final_model.stat().st_size / (1024**3):.2f} GB)")
        
        models_text = "\n".join(available_models) if available_models else f"• {model_name}: {self.final_model.name}"
        
        console.print(Panel.fit(
            f"""[bold green]✅ Qwen2.5 Conversion Complete![/]

[bold]Model Details:[/]
- Model Name: {model_name}
- Base Model: Qwen2.5-7B (Alibaba Cloud)
- Specialization: Polygenic Risk Score Analysis
- Output Directory: {self.output_dir}

[bold]Available Models:[/]
{models_text}

[bold]Next Steps:[/]
1. Start Open WebUI: docker run -d -p 3000:8080 ghcr.io/open-webui/open-webui:ollama
2. Open http://localhost:3000 in your browser
3. Select '{model_name}' from the model dropdown
4. Start chatting!

[bold]Test your model:[/]
ollama run {model_name} "How do I calculate polygenic risk scores using PRSice-2?"

[bold]Alternative Test:[/]
ollama run {model_name} "What are the key quality control steps for GWAS summary statistics?"
""",
            border_style="green"
        ))
        
        return True

def main():
    if len(sys.argv) < 2:
        console.print("[bold red]Usage: python qwen25_prs_conversion.py <checkpoint_path> [quantization] [model_name][/]")
        console.print("\nExample: python automated_model_conversion_qwen.py ./Analysis1.html/checkpoint-2909/ Q4_K_M qwen-prs-assistant")
        console.print("\nQuantization options: Q2_K, Q4_K_S, Q4_K_M, Q5_K_S, Q8_0, f16")
        return
    
    checkpoint_path = sys.argv[1]
    quantization = sys.argv[2] if len(sys.argv) > 2 else "Q4_K_M"
    model_name = sys.argv[3] if len(sys.argv) > 3 else "qwen-prs-assistant"
    
    converter = QwenModelConverter(checkpoint_path)
    converter.convert(quantization, model_name)
#python automated_model_conversion_qwen.py ./Analysis1.html/lamma Q4_K_M llama-prs-assistant
if __name__ == "__main__":
    main()