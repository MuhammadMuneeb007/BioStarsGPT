#!/usr/bin/env python3
"""
Automated Model Conversion Script
Converts your fine-tuned model to GGUF format for Open WebUI
Specialized for Polygenic Risk Score (PRS) analysis tools
"""

import os
import sys
import subprocess
import shutil
import json
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table

console = Console()

class ModelConverter:
    def __init__(self, checkpoint_path, output_dir="./gguf_output"):
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir)
        self.merged_model_path = Path("./merged_model")
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
    
    def merge_model(self):
        """Merge the LoRA adapter with the base model"""
        console.print("[bold yellow]Merging adapter with base model...[/]")
        
        try:
            from unsloth import FastLanguageModel
            import torch
            
            # Load the fine-tuned model
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(self.checkpoint_path),
                max_seq_length=4048,
                load_in_4bit=False,  # Don't use 4bit for merging
            )
            
            # Merge the adapter with base model
            model = model.merge_and_unload()
            
            # Create output directory
            self.merged_model_path.mkdir(exist_ok=True)
            
            # Save the merged model
            model.save_pretrained(str(self.merged_model_path))
            tokenizer.save_pretrained(str(self.merged_model_path))
            
            console.print(f"✅ Merged model saved to: {self.merged_model_path}")
            return True
            
        except Exception as e:
            console.print(f"[bold red]❌ Error merging model: {e}[/]")
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
                    "-DGGML_CUDA=ON"  # Updated from LLAMA_CUDA
                    # Remove -DCMAKE_CUDA_ARCHITECTURES=native as it's causing issues
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
                        "-DGGML_CUDA=OFF"  # Explicitly disable CUDA
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
        console.print(f"[bold yellow]Converting to GGUF format (quantization: {quantization})...[/]")
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        try:
            # Convert to GGUF - use --outfile instead of --outdir
            convert_script = self.llama_cpp_path / "convert_hf_to_gguf.py"
            gguf_file = self.output_dir / "model.gguf"
            
            cmd = [
                "python", str(convert_script),
                str(self.merged_model_path),
                "--outfile", str(gguf_file),
                "--outtype", "f16"
            ]
            
            console.print("Converting to GGUF format...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                console.print(f"[bold red]❌ GGUF conversion failed: {result.stderr}[/]")
                console.print(f"Command used: {' '.join(cmd)}")
                return False
            
            # Check if the GGUF file was created
            if not gguf_file.exists():
                console.print(f"[bold red]❌ GGUF file not created: {gguf_file}[/]")
                return False
            
            console.print(f"✅ GGUF file created: {gguf_file}")
            
            # Quantize the model
            if quantization != "f16":
                console.print(f"Quantizing to {quantization}...")
                quantized_file = self.output_dir / f"model-{quantization.lower()}.gguf"
                
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
                        return False
                
                cmd = [
                    str(quantize_binary),
                    str(gguf_file),
                    str(quantized_file),
                    quantization
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    console.print(f"[bold red]❌ Quantization failed: {result.stderr}[/]")
                    return False
                
                console.print(f"✅ Quantized model created: {quantized_file}")
                self.final_model = quantized_file
            else:
                self.final_model = gguf_file
            
            return True
            
        except Exception as e:
            console.print(f"[bold red]❌ Error converting to GGUF: {e}[/]")
            return False
    
    def create_modelfile(self, model_name="prs-assistant"):
        """Create Ollama Modelfile"""
        console.print("[bold yellow]Creating Ollama Modelfile...[/]")
        
        modelfile_content = f'''FROM ./{self.final_model.name}

TEMPLATE """<|system|>
{{{{ .System }}}}<|end|>
<|user|>
{{{{ .Prompt }}}}<|end|>
<|assistant|>
"""

SYSTEM """You are a helpful AI assistant specialized in polygenic risk score (PRS) analysis and related genomic tools. You provide clear, accurate, and practical information about:

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
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER stop "<|end|>"
PARAMETER stop "<|system|>"
PARAMETER stop "<|user|>"
PARAMETER stop "<|assistant|>"
'''
        
        modelfile_path = self.output_dir / "Modelfile"
        with open(modelfile_path, 'w') as f:
            f.write(modelfile_content)
        
        console.print(f"✅ Modelfile created: {modelfile_path}")
        return modelfile_path
    
    def create_ollama_model(self, model_name="prs-assistant"):
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
    
    def convert(self, quantization="Q4_K_M", model_name="prs-assistant"):
        """Main conversion pipeline"""
        console.print(Panel.fit(
            "[bold cyan]Starting Model Conversion to GGUF[/]",
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
        console.print(Panel.fit(
            f"""[bold green]✅ Conversion Complete![/]

[bold]Model Details:[/]
• Model Name: {model_name}
• Quantization: {quantization}
• GGUF File: {self.final_model}
• Size: {self.final_model.stat().st_size / (1024**3):.2f} GB

[bold]Next Steps:[/]
1. Start Open WebUI: docker run -d -p 3000:8080 ghcr.io/open-webui/open-webui:ollama
2. Open http://localhost:3000 in your browser
3. Select '{model_name}' from the model dropdown
4. Start chatting!

[bold]Test your model:[/]
ollama run {model_name} "How do I calculate polygenic risk scores using PRSice-2?"
""",
            border_style="green"
        ))
        
        return True

def main():
    if len(sys.argv) < 2:
        console.print("[bold red]Usage: python automated_model_conversion.py <checkpoint_path> [quantization] [model_name][/]")
        console.print("\nExample: python automated_model_conversion.py ./outputs_llama3.3/checkpoint-500/ Q4_K_M prs-assistant")
        console.print("\nQuantization options: Q2_K, Q4_K_S, Q4_K_M, Q5_K_S, Q8_0, f16")
        return
    
    checkpoint_path = sys.argv[1]
    quantization = sys.argv[2] if len(sys.argv) > 2 else "Q4_K_M"
    model_name = sys.argv[3] if len(sys.argv) > 3 else "prs-assistant"
    
    converter = ModelConverter(checkpoint_path)
    converter.convert(quantization, model_name)

if __name__ == "__main__":
    main()