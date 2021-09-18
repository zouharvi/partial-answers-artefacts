### DT-2
No, because there exist samples with same features but different outcomes:
4x (yellow, small, round) -> yes
1x (yellow, small, round) -> no
4x (yellow, large, regular -> yes
2x (yellow, large, regular) -> no

The performance of the whole tree on the training data is 81.25%.

### DT-3
It is used to reduce overfitting on the data. In practice, it removes sections of the tree that are deemed to be too specific. Subtrees are replaced with leaf nodes which increases error on training data but hopefully reduces generalization error.

### DT-4
`max_depth`: maximum depth to which to grow the tree
`min_samples_split`: number of samples needed to be present in a node to split it
`min_samples_leaf`: minimum number of samples needed to be present in a leaf
 
### DT-5
E.g. by setting `max_level=2` we stop after the second level of splitting (or prune the internal nodes - replace with leaves with MCC prediction). This would result in lower performance on the training data (68.75%).
In this case, however, the number of samples and features is very low and it is hard to determine whether overfitting happened in this case. To answer this question, one could do a LOOCV.
