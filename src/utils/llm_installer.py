"""
LLM Installation Utility

Provides functions to install LLM dependencies programmatically.
"""

import subprocess
import sys
from typing import Dict, Any, Optional
from pathlib import Path


def install_llm_dependencies(
    venv_path: Optional[Path] = None,
    progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Install LLM dependencies using pip.
    
    Args:
        venv_path: Optional path to virtual environment (if None, uses current environment)
        progress_callback: Optional callback function(status_message: str) for progress updates
    
    Returns:
        Dictionary with installation result:
        - success: bool - Whether installation succeeded
        - message: str - Status message
        - packages_installed: list - List of packages that were installed
        - errors: list - List of error messages if any
    """
    result = {
        "success": False,
        "message": "",
        "packages_installed": [],
        "errors": []
    }
    
    # LLM dependencies to install
    llm_packages = [
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "bitsandbytes>=0.41.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0"
    ]
    
    # Determine pip command
    if venv_path:
        if sys.platform == "win32":
            pip_cmd = str(venv_path / "Scripts" / "pip.exe")
        else:
            pip_cmd = str(venv_path / "bin" / "pip")
    else:
        pip_cmd = [sys.executable, "-m", "pip"]
    
    def update_progress(msg: str):
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)
    
    try:
        update_progress("Installing LLM dependencies...")
        
        # Install PyTorch first (if not already installed)
        try:
            import torch
            update_progress("PyTorch already installed, skipping...")
        except ImportError:
            update_progress("Installing PyTorch...")
            # Check for CUDA support
            try:
                import subprocess as sp
                cuda_check = sp.run(['nvidia-smi'], capture_output=True, timeout=5)
                has_cuda = cuda_check.returncode == 0
            except Exception:
                has_cuda = False
            
            if has_cuda:
                update_progress("NVIDIA GPU detected, installing PyTorch with CUDA...")
                cmd = [pip_cmd] if isinstance(pip_cmd, str) else pip_cmd
                cmd.extend(["install", "torch", "torchvision", "torchaudio", 
                           "--index-url", "https://download.pytorch.org/whl/cu128"])
                install_result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout
                )
                if install_result.returncode != 0:
                    # Fallback to CPU version
                    update_progress("CUDA installation failed, falling back to CPU version...")
                    cmd = [pip_cmd] if isinstance(pip_cmd, str) else pip_cmd
                    cmd.extend(["install", "torch", "torchvision", "torchaudio"])
                    install_result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=600
                    )
            else:
                update_progress("No GPU detected, installing CPU-only PyTorch...")
                cmd = [pip_cmd] if isinstance(pip_cmd, str) else pip_cmd
                cmd.extend(["install", "torch", "torchvision", "torchaudio"])
                install_result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600
                )
            
            if install_result.returncode != 0:
                result["errors"].append(f"PyTorch installation failed: {install_result.stderr}")
                result["message"] = "Failed to install PyTorch"
                return result
        
        # Install LLM packages
        for package in llm_packages:
            update_progress(f"Installing {package}...")
            cmd = [pip_cmd] if isinstance(pip_cmd, str) else pip_cmd
            cmd.extend(["install", package])
            
            install_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout per package
            )
            
            if install_result.returncode == 0:
                result["packages_installed"].append(package)
            else:
                error_msg = f"Failed to install {package}: {install_result.stderr}"
                result["errors"].append(error_msg)
                update_progress(f"Warning: {error_msg}")
        
        # Verify installation
        update_progress("Verifying installation...")
        try:
            import transformers
            import accelerate
            import torch
            result["success"] = True
            result["message"] = f"Successfully installed {len(result['packages_installed'])} packages"
            update_progress("LLM installation complete!")
        except ImportError as e:
            result["errors"].append(f"Verification failed: {str(e)}")
            result["message"] = "Installation completed but verification failed"
        
    except subprocess.TimeoutExpired:
        result["errors"].append("Installation timed out")
        result["message"] = "Installation timed out - please try again"
    except Exception as e:
        result["errors"].append(f"Unexpected error: {str(e)}")
        result["message"] = f"Installation failed: {str(e)}"
    
    return result


def check_llm_installation() -> Dict[str, Any]:
    """
    Check if LLM dependencies are installed.
    
    Returns:
        Dictionary with installation status
    """
    from .llm_check import get_llm_status
    return get_llm_status()

