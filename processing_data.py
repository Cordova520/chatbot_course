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
import processing_words as pw
from processing_words import Vocabulary

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

#Splits each lie of the file into a dictionary of fields (lineID, movieID, character, text)
line_fields = ["lineID", "characterID", "movieID", "character", "text"]
lines = {}
with open(lines_filepath, 'r', encoding='iso-8859-1') as file:
    for line in file:
        values = line.split(" +++$+++ ")
        #Extract fields
        lineObj = {}
        for i, field in enumerate(line_fields):
            lineObj[field] = values[i]
        lines[lineObj['lineID']] = lineObj

#Groups fields of lines from 'Loadlines' into conversations based on "movie_conversations.txt"
conv_fields = ["characterID", "character2ID", "movieID", "utteranceIDs"]
conversations = []
with open(conv_filepath, 'r', encoding='iso-8859-1') as file:
    for line in file:
        values = line.split(" +++$+++ ")
        #Extract fields
        convObj = {}
        for i, field in enumerate(conv_fields):
            convObj[field] = values[i]
        #Convert string result from split to list, since convObj["utteranceIDs"] == "['L98485', 'l598486', ...]"
        lineIds = eval(convObj["utteranceIDs"])
        #Reassemble lines
        convObj["lines"] = []
        for lineId in lineIds:
            convObj["lines"].append(lines[lineId])
        conversations.append(convObj)

#Extract pairs of sentences from conversations
qa_pairs = []
for conversation in conversations:
    #Iterate over all the lines of the conversation
    for i in range(len(conversation["lines"]) - 1):
        inputline = conversation["lines"][i]["text"].strip()
        targetline = conversation["lines"][i+1]["text"].strip()
        #Filter wrong samples (if one of the lists is empty)
        if inputline and targetline:
            qa_pairs.append([inputline, targetline])

#Define path to new file
datafile = os.path.join("cornell_movie_dialogs_corpus/cornell movie-dialogs corpus", "formatted_movie_lines.txt")
delimiter = '\t'
#Unescape the delimiter
delimiter = str(codecs.decode(delimiter, "unicode_escape"))

#Write new csv file
print("\nWriting newly formatted file...")
with open(datafile, 'w', encoding='utf-8') as outputfile:
    writer = csv.writer(outputfile, delimiter=delimiter)
    for pair in qa_pairs:
        writer.writerow(pair)
print("Done writing to file")

#Visualize some lines
with open(datafile, 'rb') as file:
    lines = file.readlines()
for line in lines[:8]:
    print(line)

#Turn a Unicode string to plain ASCII
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

#Test the function
print(unicodeToAscii("Montréal, Françoise...."))

#Lowercase, trim white spaces, lines...etc, and remove non-letter characters.
def normalizeString(s):
    s =unicodeToAscii(s.lower().strip())
    #Replace any .!? by a whitespace + the character --> '!' = ' ! '. \1 means the first bracketed gorup --? [,!?]. r is to
    #not consider \1 as a character (r to escape a backslash). + means one or more
    s = re.sub(r"([.!?])", r" \1", s)
    #Remove a character that is not sequence of lowe or upper case letters
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    #Remove a sequence of whitespace characters
    s= re.sub(r"\s+", r" ", s).strip()
    return s

#Test the function
print(normalizeString("aa123aa!s's   dd?"))

#Read the file and split it into lines
print("Reading and processing file.... Please wait")
lines = open(datafile, encoding='utf-8').read().strip().split('\n')
#Split every line into pairs and normalize
pairs = [[normalizeString(s) for s in pair.split('\t')] for pair in lines]
print("Done reading!")
voc = Vocabulary("cornell movie-dialogs corpus")

#Returns True if both sentences in a pair 'p' are under the MAX_LENGTH threshold
MAX_LENGTH = 10 # Maximum sentence length to consider (max words)
def filterPair(p):
    #Input sequences need to preserve the last word for EOS token
    return len(p[0].split()) < MAX_LENGTH and len(p[1].split()) < MAX_LENGTH

#Filter pairs using filterPair condition
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

pairs = [pair for pair in pairs if len(pair) > 1]
print("There are {} pairs/conversations in the dataset".format(len(pairs)))
pairs = filterPairs(pairs)
print("After filtering, there are {} pairs/conversations".format(len(pairs)))

#Loop through each pair of and add the question and reply sentence to the vocabulary
for pair in pairs:
    voc.addSentence(pair[0])
    voc.addSentence(pair[1])
print("Counted words:", voc.num_words)
for pair in pairs[:10]:
    print(pair)

MIN_COUNT = 3 #Minimum word count threshold for trimming
def trimRareWords(voc, pairs, MIN_COUNT):
    #Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    #Filter out pairs with trimmed words
    keep_pairs =[]
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        #Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        #Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        #Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs

#Trim voc and pairs
pairs = trimRareWords(voc, pairs, MIN_COUNT)
print(pairs[1][0])


def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [pw.EOS_token]

#Test the function
print(indexesFromSentence(voc, pairs[1][0]))

#Define same samples for testing
inp = []
out = []
for pair in pairs[:10]:
    inp.append(pair[0])
    out.append(pair[1])
print(inp)
print(len(inp))
indexes = [indexesFromSentence(voc, sentence) for sentence in inp]
print(indexes)

def zeroPadding(l, fillvalue = 0):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

leng = [len(ind) for ind in indexes]
print(max(leng))

#Testing the Function
test_result = zeroPadding(indexes)
print(len(test_result)) #The max length is now the number of rows
print(test_result)

def binaryMatrix(l, value=0):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == pw.PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

binary_result = binaryMatrix(test_result)
print(binary_result)

#Returns padded input sequence tensor and as well as a tensor of lengths for each of of the sequences in the batch
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVAr = torch.LongTensor(padList)
    return padVAr, lengths

#Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

#Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    #Sort the questions in descending length
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    #assert len(inp) == Lengths[0]
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len

#Example for validation
small_batch_size = 5
batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches

print("input_variable:")
print(input_variable)
print("lengths: ", lengths)
print("Target_variable:")
print(target_variable)
print("mask:")
print(mask)
print("max_target_len:", max_target_len)
