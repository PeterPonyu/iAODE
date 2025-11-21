"""
Shared utilities for iAODE examples
"""
from pathlib import Path
import os

# ANSI color codes for pretty terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text, width=70):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*width}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(width)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*width}{Colors.ENDC}\n")

def print_section(text):
    """Print a section marker"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}▸ {text}{Colors.ENDC}")

def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")

def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")

def print_info(text):
    """Print info message"""
    print(f"{Colors.CYAN}ℹ {text}{Colors.ENDC}")

def check_iaode_installed():
    """Check if iaode is installed and provide installation instructions"""
    try:
        import iaode
        print_success(f"iAODE version {iaode.__version__} detected")
        return True
    except ImportError:
        print_error("iAODE package not found!")
        print_info("Please install iAODE before running this example:")
        print(f"\n  {Colors.BOLD}pip install iaode{Colors.ENDC}")
        print("\nOr for development installation:")
        print(f"  {Colors.BOLD}pip install -e .{Colors.ENDC}")
        print()
        return False

def check_data_files(files_dict):
    """
    Check if required data files exist
    
    Parameters
    ----------
    files_dict : dict
        Dictionary mapping file descriptions to Path objects
    
    Returns
    -------
    bool
        True if all files exist
    """
    all_exist = True
    for desc, filepath in files_dict.items():
        if not filepath.exists():
            all_exist = False
            print_error(f"Missing {desc}: {filepath}")
    
    return all_exist

def setup_output_dir(example_name: str) -> Path:
    """
    Create and return output directory for example.

    Environment variables to customize behavior:
    - IAODE_OUTPUT_ROOT: Root directory for outputs (default: examples/outputs)
    - IAODE_OUTPUT_FLAT: If set to '1' or 'true', do not create per-example
      subdirectories; write all outputs directly under the root.

    Parameters
    ----------
    example_name : str
        Name of the example (e.g., 'basic_usage')

    Returns
    -------
    Path
        Output directory path
    """
    default_root = Path(__file__).parent / "outputs"
    # Support both IAODE_OUTPUT_DIR (alias) and IAODE_OUTPUT_ROOT
    root_env = os.environ.get("IAODE_OUTPUT_DIR") or os.environ.get("IAODE_OUTPUT_ROOT")
    root = Path(root_env) if root_env else default_root
    flat = os.environ.get("IAODE_OUTPUT_FLAT", "0").strip().lower() in {"1", "true", "yes"}

    output_dir = root if flat else (root / example_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
