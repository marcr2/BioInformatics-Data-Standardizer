"""
Local LLM Client for BIDS

Provides unified interfaces for local LLM inference:
- Format identification and diagnostics
- Code generation for data fixing
- 100% local and private - no API calls
"""

import os
import re
import json
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)


@dataclass
class LLMResponse:
    """Standardized response from any LLM."""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Any] = None


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    def generate_code(self, prompt: str, language: str = "python") -> str:
        """Generate code from the LLM, extracting just the code block."""
        pass


class LocalLLMClient(BaseLLMClient):
    """
    Local LLM client using transformers library.
    
    Optimized for 16GB VRAM GPUs:
    - Uses 4-bit quantization to fit larger models
    - Supports Llama 3.1, Mistral, and other open models
    - Prioritizes quality over speed
    """
    
    # Model recommendations for 16GB VRAM (prioritizing quality)
    # Format: (model_id, use_quantization, max_length, requires_auth)
    RECOMMENDED_MODELS = [
        # Best quality options (4-bit quantized for 16GB VRAM)
        # Note: Llama models require HuggingFace token
        ("meta-llama/Meta-Llama-3.1-8B-Instruct", True, 4096, True),  # High quality, good reasoning
        ("mistralai/Mistral-7B-Instruct-v0.3", True, 4096, False),     # Excellent code generation, no auth needed
        ("Qwen/Qwen2.5-7B-Instruct", True, 4096, False),               # Strong multilingual support, no auth needed
        # Fallback options (smaller, faster)
        ("microsoft/Phi-3-medium-4k-instruct", True, 4096, False),      # Smaller but efficient, no auth needed
    ]
    
    def __init__(
        self, 
        model_id: Optional[str] = None,
        use_quantization: bool = True,
        device_map: str = "auto",
        max_length: int = 4096,
        torch_dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize local LLM client.
        
        Args:
            model_id: HuggingFace model ID (default: auto-select best for GPU)
            use_quantization: Use 4-bit quantization to save VRAM
            device_map: Device placement strategy ("auto", "cuda", "cpu")
            max_length: Maximum context length
            torch_dtype: Torch dtype (default: float16 for GPU, float32 for CPU)
        """
        self.device_map = device_map
        self.max_length = max_length
        
        # Auto-select model if not provided
        if model_id is None:
            model_id = self._select_best_model()
        
        self.model_id = model_id
        
        # Determine dtype
        if torch_dtype is None:
            if torch.cuda.is_available():
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
        
        # Setup quantization config if needed
        quantization_config = None
        if use_quantization and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        print(f"Loading local LLM: {model_id}")
        print(f"  Quantization: {use_quantization}")
        print(f"  Device: {device_map}")
        
        # Check for HuggingFace token (needed for some models like Llama)
        hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
        tokenizer_kwargs = {"trust_remote_code": True}
        model_kwargs_base = {"trust_remote_code": True}
        
        if hf_token:
            tokenizer_kwargs["token"] = hf_token
            model_kwargs_base["token"] = hf_token
            print("  Using HuggingFace authentication token")
        elif "llama" in model_id.lower() or "meta-llama" in model_id.lower():
            print("  Warning: Llama models may require HuggingFace token.")
            print("  Set HUGGINGFACE_TOKEN environment variable if download fails.")
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                **tokenizer_kwargs
            )
        except Exception as e:
            if "authentication" in str(e).lower() or "token" in str(e).lower():
                print(f"\nError: Authentication required for {model_id}")
                print("Please set HUGGINGFACE_TOKEN environment variable.")
                print("Or use a model that doesn't require authentication.")
                raise ValueError(f"Model requires HuggingFace authentication: {e}")
            raise
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        model_kwargs = {
            **model_kwargs_base,
            "device_map": device_map,
            "dtype": torch_dtype,  # Use 'dtype' instead of deprecated 'torch_dtype'
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["low_cpu_mem_usage"] = True
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **model_kwargs
            )
        except Exception as e:
            if "authentication" in str(e).lower() or "token" in str(e).lower():
                print(f"\nError: Authentication required for {model_id}")
                print("Please set HUGGINGFACE_TOKEN environment variable.")
                print("Or use a model that doesn't require authentication.")
                raise ValueError(f"Model requires HuggingFace authentication: {e}")
            raise
        
        # Create pipeline for easier generation
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map=device_map,
            dtype=torch_dtype,  # Use 'dtype' instead of deprecated 'torch_dtype'
            max_length=max_length,
        )
        
        print("Local LLM loaded successfully!")
    
    def _select_best_model(self) -> str:
        """Select the best model based on available resources."""
        # Check for HuggingFace token
        hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
        prefer_no_auth = not hf_token
        
        # Check GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            print(f"Detected GPU memory: {gpu_memory:.1f} GB")
            
            # Select model based on GPU memory and auth availability
            if gpu_memory >= 14:
                # Prefer models that don't require auth if no token available
                if prefer_no_auth:
                    return self.RECOMMENDED_MODELS[1][0]  # Mistral 7B (no auth)
                else:
                    return self.RECOMMENDED_MODELS[0][0]  # Llama 3.1 8B (requires auth)
            elif gpu_memory >= 8:
                return self.RECOMMENDED_MODELS[1][0]  # Mistral 7B
            else:
                return self.RECOMMENDED_MODELS[3][0]  # Phi-3
        else:
            # CPU fallback - use smaller model that doesn't require auth
            print("No GPU detected, using CPU (this will be slow)")
            return self.RECOMMENDED_MODELS[3][0]  # Phi-3
    
    def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> LLMResponse:
        """
        Generate a response using the local LLM.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system instructions
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            top_p: Nucleus sampling parameter
            
        Returns:
            LLMResponse with the generated content
        """
        # Clear CUDA cache before generation to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Format prompt based on model type
        if "llama" in self.model_id.lower():
            formatted_prompt = self._format_llama_prompt(prompt, system_prompt)
        elif "mistral" in self.model_id.lower():
            formatted_prompt = self._format_mistral_prompt(prompt, system_prompt)
        elif "qwen" in self.model_id.lower():
            formatted_prompt = self._format_qwen_prompt(prompt, system_prompt)
        else:
            # Generic format
            if system_prompt:
                formatted_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            else:
                formatted_prompt = f"User: {prompt}\n\nAssistant:"
        
        try:
            # Generate with memory-efficient settings
            outputs = self.pipeline(
                formatted_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                return_full_text=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            
            # Extract generated text
            generated_text = outputs[0]["generated_text"].strip()
            
            # Estimate token usage (approximate)
            input_tokens = len(self.tokenizer.encode(formatted_prompt))
            output_tokens = len(self.tokenizer.encode(generated_text))
            
            result = LLMResponse(
                content=generated_text,
                model=self.model_id,
                usage={
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                },
                raw_response=outputs
            )
        finally:
            # Clear CUDA cache after generation to free up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return result
    
    def _format_llama_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Format prompt for Llama 3.1 models."""
        if system_prompt:
            return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    def _format_mistral_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Format prompt for Mistral models."""
        if system_prompt:
            return f"<s>[INST] {system_prompt}\n\n{prompt} [/INST]"
        else:
            return f"<s>[INST] {prompt} [/INST]"
    
    def _format_qwen_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Format prompt for Qwen models."""
        if system_prompt:
            return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    def generate_code(self, prompt: str, language: str = "python") -> str:
        """
        Generate code and extract it from markdown code blocks.
        
        Args:
            prompt: The prompt describing what code to generate
            language: Programming language (default: python)
            
        Returns:
            Extracted code string
        """
        system_prompt = f"""You are an expert {language} developer specializing in data transformation and cleaning.
Generate production-quality, well-documented code.
Return ONLY the code inside a single code block, no explanations before or after.
The code must be complete, handle edge cases, and include error handling."""
        
        response = self.generate(
            prompt, 
            system_prompt=system_prompt,
            temperature=0.3,  # Lower temperature for code generation
            max_new_tokens=4096
        )
        return self._extract_code_block(response.content, language)
    
    def _extract_code_block(self, text: str, language: str) -> str:
        """Extract code from markdown code blocks."""
        # Try to find code block with language specifier
        pattern = rf"```{language}\s*(.*?)```"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Try generic code block
        pattern = r"```\s*(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Return as-is if no code block found
        return text.strip()
    
    def identify_file_format(self, header_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Use local LLM to identify an unknown file format.
        
        Args:
            header_bytes: First N bytes of the file
            filename: Original filename
            
        Returns:
            Dict with format info and parsing suggestion
        """
        # Convert bytes to hex representation for safety
        hex_header = header_bytes[:256].hex()
        printable = ''.join(chr(b) if 32 <= b < 127 else '.' for b in header_bytes[:256])
        
        prompt = f"""Analyze this file and identify its format.

Filename: {filename}
File header (hex): {hex_header}
File header (printable): {printable}

Respond in this exact JSON format:
{{
    "format": "detected format name",
    "mime_type": "mime/type",
    "confidence": 0.0-1.0,
    "is_tabular": true/false,
    "parsing_hint": "brief hint on how to parse",
    "python_snippet": "pandas code to read this file or null"
}}"""
        
        system_prompt = "You are a file format identification expert. Analyze file headers and identify formats accurately. Always respond with valid JSON only."
        
        response = self.generate(prompt, system_prompt=system_prompt, temperature=0.1)
        
        # Parse JSON response
        try:
            # Extract JSON from response
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            # Try to find JSON object
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)
            
            return json.loads(content)
        except json.JSONDecodeError:
            return {
                "format": "unknown",
                "mime_type": "application/octet-stream",
                "confidence": 0.0,
                "is_tabular": False,
                "parsing_hint": "Could not identify format",
                "python_snippet": None
            }
    
    def diagnose_data(self, df_info: str, schema_rules: str, failed_scripts: List[str]) -> Dict[str, Any]:
        """
        Diagnose data quality issues.
        
        Args:
            df_info: String representation of DataFrame info
            schema_rules: Target schema validation rules
            failed_scripts: List of previously failed fix scripts to avoid
            
        Returns:
            Dict with diagnosis results
        """
        failed_context = ""
        if failed_scripts:
            failed_context = f"""
IMPORTANT: These fix approaches have FAILED before - do NOT suggest similar fixes:
{chr(10).join(f'- {script[:200]}...' for script in failed_scripts[:5])}
"""
        
        prompt = f"""Analyze this data for quality issues against the target schema.

DATA INFO:
{df_info}

TARGET SCHEMA RULES:
{schema_rules}
{failed_context}

Respond in this exact JSON format:
{{
    "is_valid": true/false,
    "overall_quality_score": 0.0-1.0,
    "issues": [
        {{
            "column": "column_name",
            "issue_type": "missing|invalid_type|out_of_range|format_error|semantic_error",
            "severity": "critical|warning|info",
            "description": "what's wrong",
            "affected_rows": "count or 'all'",
            "suggested_fix": "brief fix description"
        }}
    ],
    "summary": "overall assessment"
}}"""
        
        system_prompt = "You are a data quality expert. Analyze dataframes for issues and provide structured JSON responses. Be thorough and accurate."
        
        response = self.generate(prompt, system_prompt=system_prompt, temperature=0.2)
        
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            # Try to find JSON object
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)
            
            return json.loads(content)
        except json.JSONDecodeError:
            return {
                "is_valid": False,
                "overall_quality_score": 0.0,
                "issues": [{"column": "unknown", "issue_type": "parse_error", 
                           "severity": "critical", "description": "Failed to analyze data",
                           "affected_rows": "unknown", "suggested_fix": "Manual review required"}],
                "summary": "Analysis failed"
            }
    
    def generate_fix_script(
        self, 
        df_sample: str, 
        diagnosis: Dict[str, Any],
        target_schema: Dict[str, Any],
        successful_scripts: List[str]
    ) -> str:
        """
        Generate a Python script to fix data issues.
        
        Args:
            df_sample: Sample of the DataFrame as string
            diagnosis: Diagnosis results from DiagnosticAgent
            target_schema: Target output schema
            successful_scripts: List of previously successful scripts for reference
            
        Returns:
            Python code string that fixes the issues
        """
        examples = ""
        if successful_scripts:
            examples = f"""
REFERENCE - These similar fixes worked before:
```python
{successful_scripts[0][:1000]}
```
"""
        
        issues_desc = "\n".join(
            f"- {issue['column']}: {issue['description']} ({issue['severity']})"
            for issue in diagnosis.get("issues", [])
        )
        
        schema_cols = "\n".join(
            f"- {col['name']} ({col['type']}): {col.get('description', 'No description')}"
            for col in target_schema.get("columns", [])
        )
        
        prompt = f"""Generate a Python function to fix this data and transform it to the target schema.

INPUT DATA SAMPLE:
{df_sample}

DIAGNOSED ISSUES:
{issues_desc}

TARGET SCHEMA:
{schema_cols}
{examples}

Requirements:
1. Create a function `fix_dataframe(df: pd.DataFrame) -> pd.DataFrame`
2. Handle ALL diagnosed issues
3. Map/rename columns to match target schema
4. Convert data types appropriately
5. Handle missing values gracefully
6. Return the cleaned DataFrame

Generate ONLY the Python code, starting with necessary imports."""
        
        return self.generate_code(prompt)


# Factory functions for backward compatibility
def get_gemini_client(
    model: Optional[str] = None,
    use_gpu: Optional[bool] = None
) -> LocalLLMClient:
    """
    Factory function to get a local LLM client (replaces Gemini).
    
    Args:
        model: Optional model ID to use
        use_gpu: Whether to use GPU (None = auto-detect from preferences)
    
    Returns:
        LocalLLMClient instance
    """
    # Get GPU preference if not explicitly set
    if use_gpu is None:
        try:
            from .preferences import get_preferences
            prefs = get_preferences()
            use_gpu = prefs.is_gpu_enabled()
        except Exception:
            # Fallback to auto-detection
            use_gpu = torch.cuda.is_available()
    
    # Set device_map based on GPU preference
    if use_gpu and torch.cuda.is_available():
        device_map = "auto"
    else:
        device_map = "cpu"
    
    return LocalLLMClient(
        model_id=model,
        device_map=device_map
    )


def get_claude_client(
    model: Optional[str] = None,
    use_gpu: Optional[bool] = None
) -> LocalLLMClient:
    """
    Factory function to get a local LLM client (replaces Claude).
    
    Args:
        model: Optional model ID to use
        use_gpu: Whether to use GPU (None = auto-detect from preferences)
    
    Returns:
        LocalLLMClient instance
    """
    # Get GPU preference if not explicitly set
    if use_gpu is None:
        try:
            from .preferences import get_preferences
            prefs = get_preferences()
            use_gpu = prefs.is_gpu_enabled()
        except Exception:
            # Fallback to auto-detection
            use_gpu = torch.cuda.is_available()
    
    # Set device_map based on GPU preference
    if use_gpu and torch.cuda.is_available():
        device_map = "auto"
    else:
        device_map = "cpu"
    
    return LocalLLMClient(
        model_id=model,
        device_map=device_map
    )


# Alias for convenience
GeminiClient = LocalLLMClient
ClaudeClient = LocalLLMClient
