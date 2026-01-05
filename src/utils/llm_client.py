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
import time
from datetime import datetime
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

from .logger import get_logger, debug, info, warning


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
        
        # Measure and store model GPU memory usage
        self._model_memory_gb = None
        if torch.cuda.is_available() and device_map != "cpu":
            try:
                # Synchronize to ensure all operations are complete
                torch.cuda.synchronize()
                # Get memory allocated by the model (this includes model weights and activations)
                model_memory_bytes = torch.cuda.memory_allocated(0)
                self._model_memory_gb = model_memory_bytes / (1024**3)
                
                # Register model memory with preferences for token calculations
                try:
                    from .preferences import get_preferences
                    prefs = get_preferences()
                    prefs.register_model_memory(self.model_id, self._model_memory_gb)
                except Exception:
                    pass  # Silently fail if preferences not available
                
                total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                base_percentage = (self._model_memory_gb / total_gpu_memory) * 100
                print(f"Model GPU memory usage: {self._model_memory_gb:.2f} GB ({base_percentage:.1f}% of GPU)")
            except Exception as e:
                print(f"Warning: Could not measure model memory: {e}")
        
        print("Local LLM loaded successfully!")
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text string.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated number of tokens
        """
        if not text:
            return 0
        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            # Fallback: rough estimate of ~4 chars per token
            return len(text) // 4
    
    def get_model_memory_usage(self) -> Optional[float]:
        """
        Get the GPU memory usage of the loaded model in GB.
        
        Returns:
            Model memory usage in GB, or None if not on GPU or not measured
        """
        return self._model_memory_gb
    
    def get_max_tokens_for_generation(self) -> int:
        """
        Get the max tokens for generation from preferences.
        
        Returns:
            Max tokens to use for generation
        """
        try:
            from .preferences import get_preferences
            prefs = get_preferences()
            return prefs.get_max_tokens()
        except Exception:
            # Fallback if preferences not available
            return 4096
    
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
        start_time = time.time()
        debug(f"LLM generate called: prompt length={len(prompt)}, max_tokens={max_new_tokens}, temp={temperature}", context="LLMClient")
        
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
            
            elapsed = time.time() - start_time
            debug(f"LLM generation complete: {output_tokens} output tokens in {elapsed:.2f}s ({output_tokens/elapsed:.1f} tokens/s)", context="LLMClient")
            
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
        debug(f"Generating {language} code: prompt length={len(prompt)}", context="LLMClient")
        
        system_prompt = f"""You are an expert {language} developer specializing in data transformation and cleaning.
Generate production-quality, well-documented code.
Return ONLY the code inside a single code block, no explanations before or after.
The code must be complete, handle edge cases, and include error handling.

IMPORTANT: Keep code concise and focused. Prioritize functionality over verbosity.
Ensure the code is complete and executable - include all necessary return statements and close all brackets."""
        
        response = self.generate(
            prompt, 
            system_prompt=system_prompt,
            temperature=0.3,  # Lower temperature for code generation
            max_new_tokens=self.get_max_tokens_for_generation()
        )
        code = self._extract_code_block(response.content, language)
        debug(f"Code generated: {len(code)} chars", context="LLMClient")
        return code
    
    def _extract_code_block(self, text: str, language: str) -> str:
        """Extract code from markdown code blocks."""
        # Try to find code block with language specifier
        pattern = rf"```{language}\s*(.*?)```"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            code = match.group(1).strip()
        else:
            # Try generic code block
            pattern = r"```\s*(.*?)```"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                code = match.group(1).strip()
            else:
                # Return as-is if no code block found
                code = text.strip()
        
        # Check if code appears truncated (incomplete)
        if self._is_code_truncated(code, language):
            warning(f"Generated code appears truncated/incomplete ({len(code)} chars)", context="LLMClient")
        
        return code
    
    def _is_code_truncated(self, code: str, language: str = "python") -> bool:
        """
        Detect if code appears to be truncated/incomplete.
        
        Args:
            code: Code string to check
            language: Programming language
            
        Returns:
            True if code appears truncated
        """
        if not code:
            return True
        
        # For Python, check for common signs of truncation
        if language == "python":
            # Check if function definition exists but no return statement
            has_function_def = bool(re.search(r'def\s+\w+\s*\(', code))
            has_return = 'return' in code.lower()
            
            # If there's a function but no return, might be truncated
            if has_function_def and not has_return:
                # But could be valid if function modifies df in-place
                # Check if df is returned or modified
                if 'return df' not in code and 'return result' not in code:
                    # Might be truncated, but also might be in-place modification
                    # Check for incomplete lines (ends with ... or incomplete operators)
                    if code.rstrip().endswith('...') or code.rstrip().endswith('~'):
                        return True
            
            # Check for unbalanced brackets/parentheses (simple heuristic)
            open_parens = code.count('(')
            close_parens = code.count(')')
            open_brackets = code.count('[')
            close_brackets = code.count(']')
            open_braces = code.count('{')
            close_braces = code.count('}')
            
            # If significantly unbalanced, might be truncated
            if abs(open_parens - close_parens) > 2 or abs(open_brackets - close_brackets) > 2:
                return True
            
            # Check if ends with incomplete statement (ends with operator, dot, etc.)
            last_line = code.strip().split('\n')[-1].strip()
            if last_line and last_line[-1] in ['.', '~', '=', '+', '-', '*', '/', '&', '|', '(', '[', '{']:
                return True
        
        return False
    
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
        
        # Use higher token limit for diagnosis to ensure complete JSON response
        max_tokens = self.get_max_tokens_for_generation()
        # Ensure at least 4096 tokens for diagnosis (diagnosis can be detailed)
        diagnosis_max_tokens = max(4096, min(max_tokens, 8192))
        
        # #region agent log
        import json
        try:
            with open(r'c:\Users\marce\Desktop\BIDS\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"id":f"log_{int(datetime.now().timestamp()*1000)}_P","timestamp":int(datetime.now().timestamp()*1000),"location":"llm_client.py:600","message":"Before LLM diagnosis call","data":{"prompt_length":len(prompt),"df_info_length":len(df_info),"max_tokens":diagnosis_max_tokens,"hypothesisId":"P"},"sessionId":"debug-session","runId":"run2"}) + '\n')
        except: pass
        # #endregion
        
        # Limit prompt size to prevent OOM - if prompt is too large, truncate df_info
        MAX_PROMPT_SIZE = 50000  # Limit to ~50k chars to prevent OOM
        if len(prompt) > MAX_PROMPT_SIZE:
            warning(f"Prompt too large ({len(prompt)} chars), truncating df_info to prevent OOM", context="LLMClient")
            # Truncate df_info proportionally
            target_df_info_size = MAX_PROMPT_SIZE - len(prompt) + len(df_info) - 10000  # Leave room for other parts
            if target_df_info_size > 0 and len(df_info) > target_df_info_size:
                df_info = df_info[:target_df_info_size] + "\n\n[Data truncated to prevent OOM - showing first portion only]"
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
        
        info(f"Running diagnosis with max_tokens={diagnosis_max_tokens}, prompt_length={len(prompt)}", context="LLMClient")
        response = self.generate(
            prompt, 
            system_prompt=system_prompt, 
            temperature=0.2,
            max_new_tokens=diagnosis_max_tokens
        )
        
        try:
            content = response.content
            debug(f"LLM diagnosis response length: {len(content)} chars", context="LLMClient")
            
            # Try to extract JSON from markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            # Try to find JSON object
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)
            
            # Try to parse JSON
            result = json.loads(content)
            
            # Validate result structure
            if not isinstance(result, dict):
                raise ValueError("Result is not a dictionary")
            
            # Ensure required fields exist
            if "is_valid" not in result:
                result["is_valid"] = False
            if "overall_quality_score" not in result:
                result["overall_quality_score"] = 0.0
            if "issues" not in result:
                result["issues"] = []
            if "summary" not in result:
                result["summary"] = "Analysis completed"
            
            info(f"Diagnosis parsed successfully: {len(result.get('issues', []))} issues found", context="LLMClient")
            return result
            
        except json.JSONDecodeError as e:
            error(f"Failed to parse LLM diagnosis response as JSON: {e}", context="LLMClient")
            warning(f"Raw LLM response (first 500 chars): {content[:500]}", context="LLMClient")
            
            # Try to extract partial information from the response
            issues_found = []
            if "issue" in content.lower() or "problem" in content.lower():
                # Try to find any issues mentioned in the text
                issues_found.append({
                    "column": "unknown",
                    "issue_type": "parse_error",
                    "severity": "critical",
                    "description": "LLM response could not be parsed as JSON. The model may have provided analysis in text format instead.",
                    "affected_rows": "unknown",
                    "suggested_fix": "Check the processing log for the raw LLM response"
                })
            
            return {
                "is_valid": False,
                "overall_quality_score": 0.0,
                "issues": issues_found if issues_found else [{
                    "column": "unknown", 
                    "issue_type": "parse_error", 
                    "severity": "critical", 
                    "description": f"Failed to parse LLM response as JSON. Error: {str(e)[:100]}",
                    "affected_rows": "unknown", 
                    "suggested_fix": "The LLM may have returned text instead of JSON. Check logs for raw response."
                }],
                "summary": f"Analysis failed: JSON parse error - {str(e)[:100]}"
            }
        except Exception as e:
            error(f"Unexpected error parsing diagnosis response: {e}", context="LLMClient")
            warning(f"Raw LLM response (first 500 chars): {content[:500] if 'content' in locals() else 'N/A'}", context="LLMClient")
            return {
                "is_valid": False,
                "overall_quality_score": 0.0,
                "issues": [{"column": "unknown", "issue_type": "parse_error", 
                           "severity": "critical", "description": f"Unexpected error: {str(e)[:100]}",
                           "affected_rows": "unknown", "suggested_fix": "Check logs for details"}],
                "summary": f"Analysis failed: {str(e)[:100]}"
            }
    
    def generate_fix_script(
        self, 
        df_sample: str, 
        diagnosis: Dict[str, Any],
        target_schema: Dict[str, Any],
        successful_scripts: List[str],
        additional_context: Optional[str] = None
    ) -> str:
        """
        Generate a Python script to fix data issues.
        
        Args:
            df_sample: Sample of the DataFrame as string
            diagnosis: Diagnosis results from DiagnosticAgent
            target_schema: Target output schema
            successful_scripts: List of previously successful scripts for reference
            additional_context: Optional additional context from user
            
        Returns:
            Python code string that fixes the issues
        """
        num_issues = len(diagnosis.get("issues", []))
        num_examples = len(successful_scripts)
        info(f"Generating fix script: {num_issues} issues to address, {num_examples} reference examples", context="LLMClient")
        debug(f"Issues: {[i.get('column', 'unknown') for i in diagnosis.get('issues', [])]}", context="LLMClient")
        
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

IMPORTANT: The following modules are already pre-loaded and available:
- pandas as pd
- numpy as np  
- re (regex)
You do NOT need to import these - they are already available. If you include import statements for these modules, they will work, but they are not necessary.

CRITICAL: When using regex patterns:
- Use raw strings (r"...") for regex patterns to avoid escape sequence errors
- Example: r'^[A-Za-z][A-Za-z0-9\\-]*$' NOT '^[A-Za-z][A-Za-z0-9\\-]*$'
- Always use raw strings when calling str.replace(), str.match(), re.sub(), etc. with regex

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
6. Return the cleaned DataFrame with: return df
7. When working with pandas Series/DataFrames, ensure you're passing Series objects, not scalars, to functions that expect iterables
8. Use raw strings (r"...") for ALL regex patterns
9. Keep code concise and focused - avoid unnecessary verbosity
10. Ensure the function is COMPLETE with a return statement at the end

CRITICAL: The code must be complete and executable. Always end with: return df

Generate ONLY the Python code. Imports are optional since pd, np, and re are pre-loaded."""
        
        # Append additional context if provided
        if additional_context:
            # Check if context contains file markers (uploaded files)
            if "--- UPLOADED FILE:" in additional_context:
                prompt += f"\n\nADDITIONAL USER CONTEXT AND UPLOADED FILES:\n"
                prompt += "The user has provided context and uploaded files. "
                prompt += "When the user references a file by name in their text context, "
                prompt += "refer to the corresponding file content below.\n"
            else:
                prompt += f"\n\nADDITIONAL USER CONTEXT:\n"
            prompt += additional_context
        
        return self.generate_code(prompt)
    
    def debug_script_execution(
        self,
        script: str,
        error_message: str,
        previous_errors: Optional[List[str]] = None
    ) -> str:
        """
        Debug a script execution error and return a fixed version.
        
        Args:
            script: The script that failed to execute
            error_message: The error message from execution
            previous_errors: List of previous error messages if this is a retry
            
        Returns:
            Fixed script that should execute without errors
        """
        previous_errors = previous_errors or []
        
        previous_context = ""
        if previous_errors:
            previous_context = f"""
PREVIOUS DEBUG ATTEMPTS:
{chr(10).join(f'Attempt {i+1} error: {err[:200]}' for i, err in enumerate(previous_errors))}

IMPORTANT: The previous fixes above did NOT work. Make sure your fix addresses the actual root cause.
"""
        
        prompt = f"""A Python script failed to execute. Fix the script so it executes without errors.

FAILED SCRIPT:
```python
{script}
```

EXECUTION ERROR:
{error_message}

{previous_context}
CRITICAL REQUIREMENTS:
1. The script MUST execute without syntax errors or runtime exceptions
2. The script MUST produce a DataFrame result (either via fix_dataframe() function, 'result' variable, or modified 'df' variable)
3. The script MUST return the DataFrame at the end
4. Keep the same logic and intent as the original script - only fix the execution errors
5. Ensure all variables are defined before use
6. Ensure all function calls use correct syntax
7. Use raw strings (r"...") for regex patterns
8. Ensure all brackets, parentheses, and quotes are properly closed

IMPORTANT: The following modules are pre-loaded: pandas as pd, numpy as np, re (regex).
You do NOT need to import these.

Fix ONLY the execution errors. Do NOT change the data transformation logic unless it's causing the error.

Return ONLY the fixed Python code, no explanations."""
        
        system_prompt = """You are a Python debugging expert. Fix script execution errors while preserving the original logic and intent. 
Focus on syntax errors, undefined variables, type errors, and other runtime exceptions. 
Return only the fixed code, no explanations."""
        
        response = self.generate(
            prompt,
            system_prompt=system_prompt,
            temperature=0.1,  # Low temperature for debugging (more deterministic)
            max_new_tokens=self.get_max_tokens_for_generation()
        )
        
        # Extract code from response
        code = self._extract_code_block(response.content, "python")
        
        # If no code block found, try to use the response as-is (might be just code)
        if not code or len(code.strip()) < 10:
            code = response.content.strip()
        
        return code
    
    def rewrite_to_schema(
        self,
        df_sample: str,
        target_schema: Dict[str, Any],
        source_columns: List[str]
    ) -> str:
        """
        Generate a Python script to structurally rewrite data to match schema.
        
        IMPORTANT: This is a STRUCTURAL rewrite only. It must NOT change data values.
        Only column renaming, reordering, and reorganization are allowed.
        Data value transformations are handled by fix scripts, not this method.
        
        Args:
            df_sample: Sample of the DataFrame as string
            target_schema: Target output schema
            source_columns: List of source column names
            
        Returns:
            Python code string that reorders/renames columns to match schema
        """
        schema_cols = "\n".join(
            f"- {col['name']} ({col['type']}): {col.get('description', 'No description')}"
            for col in target_schema.get("columns", [])
        )
        
        source_cols_str = ", ".join(source_columns)
        
        prompt = f"""Generate a Python function to STRUCTURALLY rewrite this data to match the target schema.

IMPORTANT CONSTRAINTS - READ CAREFULLY:
1. This is a STRUCTURAL REWRITE step, NOT a data fix step.
2. You are ONLY reorganizing/remapping data structure to match the schema.
3. Data value transformations (like calculating p-values, fixing errors, etc.) will be handled by SEPARATE fix scripts later.

CRITICAL RULES:
- You must preserve ALL data. Do NOT drop any rows.
- The output DataFrame must have EXACTLY the same number of rows as the input.
- **Data values themselves must NOT be changed**. Only column names, order, and organization can be modified.
- Actual cell values must remain IDENTICAL. This rewrite step is purely structural.
- ONLY perform column renaming, reordering, and reorganization.
- NO calculations, transformations, merges, splits, or any modifications to cell values.
- Map existing columns to schema columns by renaming/reordering them only.
- Do NOT modify the actual data values in cells.
- If a schema column doesn't exist in input, you may create it by copying an existing column (same values), but do NOT calculate or transform values.
- If multiple input columns could map to one schema column, choose the best match and rename it. Do NOT combine or merge column values.

AVAILABLE MODULES: pd (pandas), np (numpy), re (regex) are pre-loaded.

INPUT DATA SAMPLE:
{df_sample}

SOURCE COLUMNS:
{source_cols_str}

TARGET SCHEMA:
{schema_cols}

Requirements:
1. Create a function `rewrite_dataframe(df: pd.DataFrame) -> pd.DataFrame`
2. ONLY rename and reorder columns to match schema
3. Do NOT modify any data values - they must remain identical
4. Return the rewritten DataFrame
5. Include comments documenting which input column was renamed/mapped to which schema column

Generate ONLY the Python code. Imports are optional since pd, np, and re are pre-loaded."""
        
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
