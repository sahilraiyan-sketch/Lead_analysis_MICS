"""Paths read from the MICS_DIR and OUT_DIR environment variables."""
import os
HERE=os.path.dirname(os.path.abspath(__file__))
MICS_DIR=os.path.abspath(os.environ.get('MICS_DIR','data'))
OUT_DIR=os.path.abspath(os.environ.get('OUT_DIR','out'))
os.makedirs(OUT_DIR,exist_ok=True)
