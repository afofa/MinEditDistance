import numpy as np
from typing import Dict, Tuple

def min_edit_distance(str1:str, str2:str, is_levenshtein:bool=False) -> int:
    """
    Calculates minimum edit distance (i.e. the minimum number of editing operations: insertion, deletion, substitution) between given two strings
    see the link for details: https://web.stanford.edu/class/cs124/lec/med.pdf    
    """

    # substitution cost, 1 if each operation has the same cost and 2 if levenshtein
    sub_cost = 1 + is_levenshtein

    # lenghts of input strings
    len1, len2 = len(str1), len(str2)

    # initialize distance array
    d = np.zeros((len1+1, len2+1), dtype=np.int)

    # set distances between substrings and empty string
    d[0,:] = np.arange(0, len2+1)
    d[:,0] = np.arange(0, len1+1)

    # traverse
    for i2 in range(1, len2+1):
        for i1 in range(1, len1+1):
            # characters examined
            char1, char2 = str1[i1-1], str2[i2-1]
            # update according to dynamic programming
            d[i1,i2] = min(d[i1-1,i2]+1, d[i1,i2-1]+1, d[i1-1,i2-1]+sub_cost*(char1 != char2))

    # return minimum edit distance
    return d[-1, -1]

def min_weighted_edit_distance( str1:str, str2:str, 
                                insertion:Dict[str, float]=dict(), deletion:Dict[str, float]=dict(), substitution:Dict[Tuple[str, str], float]=dict(),
                                def_insertion:float=1, def_deletion:float=1, def_substitution:float=1) -> float:
    """
    Calculates minimum weighted edit distance (i.e. the minimum number of editing operations: insertion, deletion, substitution) between given two strings,
    weights for insertion, deletion and substitution operations are given by input variables (as dicts) insertion, deletion and substitution, respectively.
    If weight is not provided, then defaults to def_insertion, def_deletion or def_substitution, respectively.
    see the link for details: https://web.stanford.edu/class/cs124/lec/med.pdf    
    """

    # lenghts of input strings
    len1, len2 = len(str1), len(str2)

    # initialize distance array
    d = np.zeros((len1+1, len2+1), dtype=np.float)

    # set distances between substrings and empty string
    for i1 in range(1, len1+1):
        char1 = str1[i1-1]
        # add default values, if weight not provided
        if char1 not in deletion.keys():
            deletion[char1] = def_deletion
        d[i1, 0] = d[i1-1, 0] + deletion[char1]
    for i2 in range(1, len2+1):
        char2 = str2[i2-1]
        # add default values, if weight not provided
        if char2 not in insertion.keys():
            insertion[char2] = def_insertion
        d[0, i2] = d[0, i2-1] + insertion[char2]

    # traverse
    for i2 in range(1, len2+1):
        for i1 in range(1, len1+1):
            # characters examined
            char1, char2 = str1[i1-1], str2[i2-1]
            # add default values, if weight not provided
            if char1 not in deletion.keys():
                deletion[char1] = def_deletion
            if char2 not in insertion.keys():
                insertion[char2] = def_insertion
            if (char1, char2) not in substitution.keys():
                if char1 != char2:
                    substitution[(char1, char2)] = def_substitution
                else:
                    substitution[(char1, char2)] = 0
            # update according to dynamic programming
            d[i1,i2] = min(d[i1-1,i2]+deletion[char1], d[i1,i2-1]+insertion[char2], d[i1-1,i2-1]+substitution[(char1, char2)])

    # return minimum edit distance
    return d[-1, -1]

def min_edit_distance_with_backtrack(str1:str, str2:str, is_backtrack:bool=False, is_levenshtein:bool=True) -> (int, np.ndarray):
    """
    Calculates minimum edit distance (i.e. the minimum number of editing operations: insertion, deletion, substitution) between given two strings,
    with backtracking to store where we came from to each cell and corresponding operation (insertion, deletion, substitution)
    see the link for details: https://web.stanford.edu/class/cs124/lec/med.pdf    
    """

    # operations
    ops = ["DELETION", "INSERTION", "SUBSTITUTION"]

    # substitution cost, 1 if each operation has the same cost and 2 if levenshtein
    sub_cost = 1 + is_levenshtein

    # lenghts of input strings
    len1, len2 = len(str1), len(str2)

    # initialize distance and backtracking arrays
    d = np.zeros((len1+1, len2+1), dtype=np.int)
    ptr = np.zeros((len1+1, len2+1), dtype="object")

    # set distances between substrings and empty string
    d[0,:] = np.arange(0, len2+1)
    d[:,0] = np.arange(0, len1+1)
    ptr[0,1:] = ops[0]
    ptr[1:,0] = ops[1]

    # traverse
    for i2 in range(1, len2+1):
        for i1 in range(1, len1+1):
            # characters examined
            char1, char2 = str1[i1-1], str2[i2-1]
            # update according to dynamic programming
            vals = [d[i1-1,i2]+1, d[i1,i2-1]+1, d[i1-1,i2-1]+sub_cost*(char1 != char2)]
            ind_min = np.argmin(vals)
            d[i1,i2] = vals[ind_min]
            ptr[i1,i2] = ops[ind_min]

    # return minimum edit distance and backtracking array
    return d[-1, -1], ptr

word1, word2 = "Intention", "Execution"

print(min_weighted_edit_distance(word1, word2))