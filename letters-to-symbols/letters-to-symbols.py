# -*- coding: utf-8 -*-
"""letters-to-symbols.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1KQdmsE4xiVVXX8cvZ-xWHV1NXl9R6W7u
"""

def letter_to_symbols(s):
  count = 1
  result = ""
  for i in range(len(s)):
   if i + 1 < len(s) and s[i] == s[i + 1]:
    count += 1
   else:
    result += str(count) + s[i]
    count = 1

  return result

letter_to_symbols("AAAABBBCCDAAA")