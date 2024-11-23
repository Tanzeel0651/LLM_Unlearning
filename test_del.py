#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 03:57:31 2024

@author: tanzeel
"""



import matplotlib.pyplot as plt


x = [45,65,23,45,18,20,31,96,48]
plt.figure(figsize=(10, 6))
plt.plot(x, label='Training Loss', color='red', marker='o')

plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# plt.grid(True)
plt.legend()
plt.show()