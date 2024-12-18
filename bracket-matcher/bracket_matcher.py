# -*- coding: utf-8 -*-
"""bracket_matcher.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1qnGwPrsoB4WLcIRpXKFFbh78tRNzaUMV
"""



def BracketMatcher(s):
  count_open = 0
  count_close = 0
  for i in range(len(s)):
    if s[i] == '(':
      count_open += 1
    elif s[i] == ')':
      count_close += 1

  if count_open == count_close:
    return True
  else:
    return False

#BracketMatcher("(a))")
BracketMatcher("(a (( kl ( mns ) t) uvwz)")