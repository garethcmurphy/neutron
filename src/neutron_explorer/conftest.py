"""test conftest.py setup for neutron_explorer package"""
import sys
from pathlib import Path

# Add the 'src' directory to the Python path
src_path = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(src_path))
