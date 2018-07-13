from cleaning.eng_rules import hybrid_g2p, rulebased_g2p
from cleaning.cleaners import english_cleaners

def pronounce_chars(line):
    line = english_cleaners(line)
    r = hybrid_g2p(line)
    return r
