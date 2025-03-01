#!/usr/bin/env python
"""
JADE Environment Setup Script
------------------------------------------------------------
This script creates a virtual environment with optimized numerical libraries
for the JADE exoplanet simulation framework, tailored to your hardware.
------------------------------------------------------------

 usage: python setup.py [--env env_name] [--method method_name] [--force]
 
 optional arguments:
   --env env_name        environment name or path (default: .venv)
   --method method_name  installation method: auto (detect best), conda, uv, or pip (default: auto)
   --force               force recreation if environment exists

------------------------------------------------------------

 Author: Mara Attia

------------------------------------------------------------
"""

import os
import sys
import subprocess
import platform
import argparse
import time
import re
import shutil

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_header(text):
    """Print a formatted header section."""
    terminal_width = shutil.get_terminal_size().columns
    print("\n" + "═" * terminal_width)
    print(f"{Colors.BOLD}{Colors.BLUE}◆ {text}{Colors.END}")
    print("═" * terminal_width)

def print_step(text):
    """Print a step in the process."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}→ {text}{Colors.END}")

def print_info(text):
    """Print information text."""
    print(f"  {Colors.YELLOW}ℹ {text}{Colors.END}")

def print_success(text):
    """Print success message."""
    print(f"  {Colors.GREEN}✓ {text}{Colors.END}")

def print_warning(text):
    """Print warning message."""
    print(f"  {Colors.YELLOW}⚠ {text}{Colors.END}")

def print_error(text):
    """Print error message."""
    print(f"  {Colors.RED}✗ {text}{Colors.END}")

def run_command(cmd, verbose=True, check=True, capture_output=True):
    """Run a shell command and return the output."""
    if verbose:
        print(f"  {Colors.BOLD}$ {' '.join(cmd)}{Colors.END}")
    
    if capture_output:
        result = subprocess.run(cmd, capture_output=True, text=True)
    else:
        # Show output directly
        result = subprocess.run(cmd)
    
    if check and result.returncode != 0:
        print_error(f"Command failed with exit code {result.returncode}")
        if capture_output and result.stderr:
            print(f"{Colors.RED}{result.stderr}{Colors.END}")
        if not verbose:
            print_error(f"Failed command: {' '.join(cmd)}")
        return result
    
    return result

def detect_cpu():
    """Detect CPU architecture and brand."""
    print_step("Detecting system architecture")
    
    cpu_info = {}
    
    # For macOS
    if platform.system() == "Darwin":
        result = run_command(["sysctl", "-n", "machdep.cpu.brand_string"], verbose=False)
        cpu_brand = result.stdout.strip()
        
        # Check if Apple Silicon
        if platform.machine() == "arm64":
            cpu_info = {"type": "apple_silicon", "brand": "Apple Silicon"}
        else:
            cpu_info = {"type": "intel", "brand": cpu_brand}
    
    # For Linux
    elif platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        cpu_brand = re.sub(".*model name.*:", "", line, 1).strip()
                        if "AMD" in cpu_brand:
                            cpu_info = {"type": "amd", "brand": cpu_brand}
                            break
                        else:
                            cpu_info = {"type": "intel", "brand": cpu_brand}
                            break
        except:
            cpu_info = {"type": "unknown", "brand": platform.processor()}
    
    # For Windows
    elif platform.system() == "Windows":
        result = run_command(["wmic", "cpu", "get", "name"], verbose=False, check=False)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                cpu_brand = lines[1].strip()
                if "AMD" in cpu_brand:
                    cpu_info = {"type": "amd", "brand": cpu_brand}
                else:
                    cpu_info = {"type": "intel", "brand": cpu_brand}
    
    # Default fallback
    if not cpu_info:
        cpu_info = {"type": "unknown", "brand": platform.processor()}
    
    print_info(f"Detected CPU: {cpu_info['brand']}")
    print_info(f"Architecture: {cpu_info['type'].upper()}")
    
    return cpu_info

def check_uv_available():
    """Check if UV is available in the system."""
    try:
        result = run_command(["uv", "--version"], verbose=False, check=False)
        return result.returncode == 0
    except:
        return False

def install_uv():
    """Install UV package manager."""
    print_step("Installing UV package manager")
    print_info("This will install UV to your user directory")
    
    try:
        # Use curl to download and run the UV install script
        if platform.system() == "Windows":
            print_error("Automatic UV installation not supported on Windows")
            print_info("Please install UV manually: https://github.com/astral-sh/uv")
            return False
        else:
            print_info("Downloading and running UV installer...")
            curl_cmd = "curl -LsSf https://astral.sh/uv/install.sh | sh"
            subprocess.run(curl_cmd, shell=True, check=True)
            
            # Verify installation
            uv_path = os.path.expanduser("~/.cargo/bin/uv")
            if os.path.exists(uv_path):
                # Add to PATH for this session
                os.environ["PATH"] = os.path.expanduser("~/.cargo/bin") + os.pathsep + os.environ["PATH"]
                print_success("UV installed successfully!")
                return True
            else:
                print_error("UV installation failed")
                return False
    except Exception as e:
        print_error(f"UV installation failed: {str(e)}")
        return False

def run_benchmark(python_cmd):
    """Run a simple matrix multiplication benchmark to measure performance."""
    print_step("Running performance benchmark")
    
    benchmark_code = """
import numpy as np
import time

# Create large matrices
size = 2000
A = np.random.random((size, size))
B = np.random.random((size, size))

# Warm up
_ = A @ B

# Measure matrix multiplication time (3 runs)
times = []
for _ in range(3):
    start = time.time()
    C = A @ B
    end = time.time()
    times.append(end-start)

avg_time = sum(times) / len(times)
print(f"Benchmark result: {avg_time:.2f} seconds")
    """
    
    benchmark_file = "jade_benchmark.py"
    with open(benchmark_file, "w") as f:
        f.write(benchmark_code)
    
    print_info("Measuring linear algebra performance...")
    try:
        if isinstance(python_cmd, list):
            command = python_cmd + [benchmark_file]
        else:
            command = [python_cmd, benchmark_file]
            
        result = run_command(command, verbose=False)
        
        # Extract time
        match = re.search(r"Benchmark result: ([\d\.]+) seconds", result.stdout)
        if match:
            runtime = float(match.group(1))
            print_info(f"Benchmark completed in {runtime:.2f} seconds")
            
            if runtime < 2.0:
                print_success("Excellent performance!")
            elif runtime < 5.0:
                print_success("Good performance")
            else:
                print_warning(f"Performance could be better. Consider using conda with MKL.")
            
            return runtime
    except Exception as e:
        print_error(f"Benchmark failed: {str(e)}")
    finally:
        if os.path.exists(benchmark_file):
            os.remove(benchmark_file)
    
    return None

def setup_environment(env_name=".venv", method="auto", force=False):
    """Create and setup an optimized environment for JADE."""
    print('')
    print(f'{Colors.GREEN}                    ░▒▓█▓▒░░▒▓██████▓▒░░▒▓███████▓▒░░▒▓████████▓▒░              {Colors.END}')
    print(f'{Colors.GREEN}                    ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░                     {Colors.END}')
    print(f'{Colors.GREEN}                    ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░                     {Colors.END}')
    print(f'{Colors.GREEN}                    ░▒▓█▓▒░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓██████▓▒░                {Colors.END}')
    print(f'{Colors.GREEN}             ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░                     {Colors.END}')
    print(f'{Colors.GREEN}             ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░                     {Colors.END}')
    print(f'{Colors.GREEN}              ░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓███████▓▒░░▒▓████████▓▒░              {Colors.END}')
    print('')

    print_header("JADE ENVIRONMENT SETUP")
    
    # Get current directory
    current_dir = os.path.abspath(os.getcwd())
    
    # Detect CPU
    cpu_info = detect_cpu()
    
    # Determine best installation method if auto
    if method == "auto":
        if cpu_info['type'] == "intel":
            method = "conda" 
        elif cpu_info['type'] == "apple_silicon" or cpu_info['type'] == "amd":
            method = "uv"
        else:
            method = "uv"  # Default fallback
    
    print_step(f"Selected installation method: {method.upper()}")
    if method == "conda":
        print_info("Using conda with MKL for optimal performance on Intel CPUs")
    elif method == "uv":
        print_info("Using UV for precise Python version control and fast installation")
    else:
        print_info("Using pip for a standard installation")
    
    # Platform specifics
    is_windows = platform.system() == "Windows"
    
    # For conda installations, we use the environment name only (not a path)
    # For pip/uv installations, we create the environment in the current directory
    if method == "conda":
        # Simple name without path for conda to avoid space issues
        local_venv_path = None  # No local path for conda environments
        
        # Check if environment already exists in conda
        conda_info = run_command(["conda", "env", "list"], verbose=False)
        env_exists = re.search(rf"{env_name}\s", conda_info.stdout) is not None
        
        if env_exists:
            if force:
                print_step(f"Removing existing conda environment: {env_name}")
                run_command(["conda", "env", "remove", "-n", env_name], capture_output=False)
            else:
                print_error(f"Conda environment '{env_name}' already exists")
                print_info("Use --force to recreate the environment")
                return
    else:
        # For pip/venv, we use a local path in the current directory
        local_venv_path = os.path.join(current_dir, env_name)
        
        # Check if environment directory exists
        if os.path.exists(local_venv_path):
            if force:
                print_step(f"Removing existing environment: {local_venv_path}")
                try:
                    shutil.rmtree(local_venv_path)
                except Exception as e:
                    print_error(f"Failed to remove existing environment: {str(e)}")
                    sys.exit(1)
            else:
                print_error(f"Environment at {local_venv_path} already exists")
                print_info("Use --force to recreate the environment")
                return
    
    # Install based on method
    if method == "conda":
        # Check if conda is available
        print_step("Checking conda availability")
        conda_check = run_command(["conda", "--version"], verbose=False, check=False)
        if conda_check.returncode != 0:
            print_error("Conda not found")
            print_info("Please install conda first or choose a different method")
            sys.exit(1)
        
        # Create conda environment using existing file
        print_step(f"Creating conda environment: {env_name}")
        
        print_info("Using existing environment.yml file")
        print_info("Installing packages (this may take a while)...")
        result = run_command(["conda", "env", "create", "-f", "environment.yml", "-n", env_name], 
                             verbose=True, capture_output=False)
            
        if result.returncode != 0:
            print_error("Conda environment creation failed")
            sys.exit(1)
            
        print_success("Conda environment created successfully!")
        
        # Use conda run to execute benchmark
        conda_python = ["conda", "run", "-n", env_name, "python"]
        benchmark_time = run_benchmark(conda_python)
        
        # Get activation command for conda
        if is_windows:
            activate_cmd = f"conda activate {env_name}"
        else:
            activate_cmd = f"conda activate {env_name}"
        
        installation_path = "<CONDA_INSTALL_PATH>/envs/" + env_name
    
    elif method == "uv":
        # Check if UV is available
        uv_available = check_uv_available()
        
        if not uv_available:
            print_step("UV package manager not found")
            user_input = input("  Would you like to install UV now? (y/n): ").strip().lower()
            
            if user_input == 'y':
                uv_installed = install_uv()
                if not uv_installed:
                    print_warning("Falling back to standard pip installation")
                    method = "pip"
            else:
                print_info("Falling back to standard pip installation")
                method = "pip"
        
        if method == "uv":  # If we still want to use UV
            # Create virtual environment with UV and Python 3.8
            print_step(f"Creating virtual environment with UV at: {local_venv_path}")
            
            # First, install Python 3.8 if needed
            print_info("Installing Python 3.8 with UV (if not already available)...")
            run_command(["uv", "python", "install", "cpython-3.8"], capture_output=False)
            
            # Create the virtual environment with Python 3.8
            print_info("Creating virtual environment with Python 3.8...")
            run_command(["uv", "venv", "--python", "cpython-3.8", local_venv_path], 
                       capture_output=False)
            
            # Get path to the virtual environment executables
            venv_bin = os.path.join(local_venv_path, "Scripts" if is_windows else "bin")
            
            # Install dependencies with UV
            print_step("Installing dependencies with UV")
            print_info("Installing packages (this will be fast!)...")
            
            # If this is a non-Windows system, we need to use the UV from the venv
            uv_cmd = os.path.join(venv_bin, "uv")
            
            run_command([uv_cmd, "pip", "install", "-r", "requirements.txt"], 
                       capture_output=False)
            
            # Run benchmark
            python_cmd = os.path.join(venv_bin, "python")
            benchmark_time = run_benchmark(python_cmd)
            
            # Get venv activation command
            if is_windows:
                activate_cmd = f"{venv_bin}\\activate"
            else:
                activate_cmd = f"source {venv_bin}/activate"
            
            installation_path = local_venv_path
    
    else:  # pip method
        # Create virtual environment
        print_step(f"Creating virtual environment at: {local_venv_path}")
        venv_cmd = [sys.executable, "-m", "venv", local_venv_path]
        run_command(venv_cmd)
        
        # Get path to python and pip in the new venv
        venv_bin = os.path.join(local_venv_path, "Scripts" if is_windows else "bin")
        python_cmd = os.path.join(venv_bin, "python")
        pip_cmd = os.path.join(venv_bin, "pip")
        
        # Upgrade pip in the new environment
        print_step("Upgrading pip")
        run_command([python_cmd, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install numpy, scipy, etc.
        print_step("Installing dependencies")
        print_info("Installing packages (this may take a while)...")
        run_command([pip_cmd, "install", "-r", "requirements.txt"], capture_output=False)
        
        # Run benchmark
        benchmark_time = run_benchmark(python_cmd)
        
        # Get venv activation command
        if is_windows:
            activate_cmd = f"{venv_bin}\\activate"
        else:
            activate_cmd = f"source {venv_bin}/activate"
        
        installation_path = local_venv_path
    
    # Print activation instructions
    print_header("INSTALLATION COMPLETE")
    print_success(f"JADE environment successfully created: {installation_path}")
    
    print_step("Activation Instructions")
    print_info(f"To activate the environment, run:")
    print(f"\n    {Colors.BOLD}{Colors.GREEN}{activate_cmd}{Colors.END}\n")
    
    # Return activation command
    return activate_cmd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up an optimized environment for JADE")
    parser.add_argument("--env", default=".venv", help="Environment name or path (default: .venv)")
    parser.add_argument("--method", choices=["auto", "conda", "uv", "pip"], default="auto", 
                        help="Installation method: auto (detect best), conda, uv, or pip (default: auto)")
    parser.add_argument("--force", action="store_true", help="Force recreation if environment exists")
    args = parser.parse_args()
    
    try:
        setup_environment(args.env, args.method, args.force)
    except KeyboardInterrupt:
        print_error("\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"\nSetup failed: {str(e)}")
        sys.exit(1)
