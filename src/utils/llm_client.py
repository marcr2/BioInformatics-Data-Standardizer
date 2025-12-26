"""
Multi-LLM Client for BIDS

Provides unified interfaces for:
- Google Gemini (FormatScout, DiagnosticAgent)
- Anthropic Claude Opus (FixingAgent)
"""

import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import google.generativeai as genai
import anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


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


class GeminiClient(BaseLLMClient):
    """
    Google Gemini client for FormatScout and DiagnosticAgent.
    
    Used for:
    - Identifying unknown file formats
    - Diagnosing data quality issues
    - Fast, cost-effective operations
    """
    
    def __init__(self, model: str = "gemini-1.5-flash"):
        """
        Initialize Gemini client.
        
        Args:
            model: Gemini model to use (default: gemini-1.5-flash for speed)
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.model_name = model
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """
        Generate a response using Gemini.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system instructions
            
        Returns:
            LLMResponse with the generated content
        """
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        response = self.model.generate_content(full_prompt)
        
        return LLMResponse(
            content=response.text,
            model=self.model_name,
            usage={
                "prompt_tokens": response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
                "completion_tokens": response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0
            },
            raw_response=response
        )
    
    def generate_code(self, prompt: str, language: str = "python") -> str:
        """
        Generate code and extract it from markdown code blocks.
        
        Args:
            prompt: The prompt describing what code to generate
            language: Programming language (default: python)
            
        Returns:
            Extracted code string
        """
        system_prompt = f"""You are a code generation assistant. 
Generate only {language} code. 
Return ONLY the code inside a single code block, no explanations.
The code must be complete and runnable."""
        
        response = self.generate(prompt, system_prompt)
        return self._extract_code_block(response.content, language)
    
    def _extract_code_block(self, text: str, language: str) -> str:
        """Extract code from markdown code blocks."""
        import re
        
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
        Use Gemini to identify an unknown file format.
        
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
        
        response = self.generate(prompt)
        
        # Parse JSON response
        import json
        try:
            # Extract JSON from response
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
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
        
        response = self.generate(prompt)
        
        import json
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
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


class ClaudeClient(BaseLLMClient):
    """
    Anthropic Claude Opus client for FixingAgent.
    
    Used for:
    - Generating Python fix scripts
    - Complex code reasoning
    - High-quality code generation
    """
    
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        """
        Initialize Claude client.
        
        Args:
            model: Claude model to use (default: claude-sonnet-4-20250514)
        """
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """
        Generate a response using Claude.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system instructions
            
        Returns:
            LLMResponse with the generated content
        """
        messages = [{"role": "user", "content": prompt}]
        
        kwargs = {
            "model": self.model_name,
            "max_tokens": 4096,
            "messages": messages
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
        
        response = self.client.messages.create(**kwargs)
        
        return LLMResponse(
            content=response.content[0].text,
            model=self.model_name,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens
            },
            raw_response=response
        )
    
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
        
        response = self.generate(prompt, system_prompt)
        return self._extract_code_block(response.content, language)
    
    def _extract_code_block(self, text: str, language: str) -> str:
        """Extract code from markdown code blocks."""
        import re
        
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


def get_gemini_client(model: str = "gemini-1.5-flash") -> GeminiClient:
    """Factory function to get a Gemini client."""
    return GeminiClient(model)


def get_claude_client(model: str = "claude-sonnet-4-20250514") -> ClaudeClient:
    """Factory function to get a Claude client."""
    return ClaudeClient(model)
