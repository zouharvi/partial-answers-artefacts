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


### KNN-1a
Note that the figure uses scaling across features followed up with individual vector normalization.

For uniform mode, train accuracy is good with K=1 but that does not hold for the LOOCV accuracy which gets better with larger K and reaches a maximum in the interval 150-250.

For distance mode (vote weight dependent on the distance) the train accuracy is constant at 100% (because of diminishing vote weight). Interestingly, the LOOCV accuracy charasteristics are similar to those of uniform mode.

Higher K come at the cost of longer retrieval times (can be alleviated with approximate search, e.g. FAISS).

### KNN-1b
It may happen that different classes are distributed differ ently in the vector space and hence different Ks would lead to varying class performance. In most scenarios we can assume, however, that the class performance is independent on the varying values of K.

This is the case also for this data (tested in an experiment).

### KNN-1c
Too low K leads to overfitting (large variance) and too high K leads to large bias (in extreme cases, to MCCC).

### Comparison
Single-core performance with BoW vectorizer without any scaling. Train time of KNN should be taken as the time it takes to vectorize the data and process them in the memory (because vanilla KNN has no training phase).

Train:
KNN (distance, 200 neighbours): 0.38s
KNN (distance, 5 neighbours): 0.37s
Naive Bayes (complement): 0.44s
Naive Bayes (multinomial): 0.37s
Random Forest (200 estimators): 10.4s
Decision Tree: 1.4s

Inference (train data):
KNN (distance, 200 neighbours): 2.94s
KNN (distance, 5 neighbours): 2.74s
Naive Bayes (complement): 0.35s
Naive Bayes (multinomial): 0.33s
Random Forest (200 estimators): 0.77s
Decision Tree: 0.34s


### Best Model
The used model is an soft-voting ensemble of:
- Complement Naive Bayes (weight 1.3)
- Multinomial Naive Bayes (weight 0.8)
- Random Forest 200 estimators (weight 1.0)
- Distance KNN with eucledian distance, 200 neighbours (weight 0.3)
- Distance KNN with cosine distance, 200 neighbours (weight 0.3)

The model and vectorizer parameters were individually optimized with a small-scale grid search. The weights were then determined by another run of grid search. They all use TF-IDF vectorizer but with varying parameters (e.g. removing stopwords, different n_gram_range, max_df and max_features). Features for the KNN are scaled and the resulting vectors normalized.

This ensemble was chosen because it outperformed other tried models. In assignment 1, we submitted a simple Logistic Regression which had better performance than this ensemble but its submission is not allowed.

What didn't help:
- WorNetLemmatizer
- Custom features (e.g. review length)
- Feature union with BoW
- This elaborate feature kernel that was supposed to model term interactions (combined with standard TF-IDF):
```
("tfidf_kernel", Pipeline([      
    ("tfidf_crop", TfidfVectorizer(stop_words="english", max_features=1000)),
    ('kernel', sklearn.preprocessing.PolynomialFeatures(interaction_only=True, include_bias=False)),
])),
```