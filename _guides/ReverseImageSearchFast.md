---
layout: guide
title: "Reverse Image Search (Fast)"
---

How it works

1) Extract feature vectors from large set of images

2) PCA

3) Build KD-tree

Why it's useful
 - substitution
 - correlating to chains of data (text)
 - fast creation of training sets for transfer learning or new classifiers, or dynamic training sets




the process has O(nm^2) runtime and took a 2013 macbook pro around 30 hours (half feature extraction and half PCA) 
