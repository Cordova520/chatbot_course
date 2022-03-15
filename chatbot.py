import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
import itertools

CUDA = torch.cuda.is_available()
device = torch.device("cuda" if CUDA else "cpu")

#Part 1: Data Processing

lines_filepath = os.path.join("cornell_movie_dialogs_corpus/cornell movie-dialogs corpus", "movie_lines.txt")
conv_filepath = os.path.join("cornell_movie_dialogs_corpus/cornell movie-dialogs corpus", "movie_conversations.txt")
#Visualize some lines
with open(lines_filepath, 'r', encoding='iso-8859-1') as file:
    lines = file.readlines()
for line in lines[:8]:
    print(line.strip())
