"""Main function prototype
"""
# Coding: utf-8
# Filename: mane.py
# Created: 2016-08-08
# Description:
## v0.0: File created.

import argparse

def parse_args():
  """
  Parse argument for easier testing.
  """
  parser = argparse.ArgumentParser(description="Train mane model.")
  parser.add_argument('--graph_pname', nargs='?', 
                      default='data/blogcatalog.graph',
                      help='graph data pickle file')
  parser.add_argument('--output', nargs='?',
                      default='emb/blogcatalog.emb',
                      help='embedding output location')
  parser.add_argument('--emb_dim', nargs='?', type=int,
                      default='128', help='Embedding dimension')
  parser.add_argument('--walk_length', nargs='?', type=int,
                      default=10, help='Length of each walk')
  parser.add_argument('--window_size', nargs='?', type=int,
                      default=10, help='
