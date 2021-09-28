## Assignment Questions

### NN-1

The output class is distributed as follows:

Class | Frequency
-|-
CARDINAL | 14.88%
DATE     |  7.93%
GPE      | 17.97%
ORG      | 33.51%
PERSON   | 25.71%

Thus, the training accuracy is `33.51%` by always predicting `ORG`.

### NN-2

Dropout creates a filter that is applied element-wise.
With probability _p_ the original value (output of the previous layer) is replaced with 0, which prevents overfitting.
It can be compared to bagging because of the model averaging, though in case of dropout, this happens in the space of a single neuron.
During inference this probabilistic filter is not applied.
This creates a possible issue because the input distribution is suddenly higher (e.g. previous average to one neuron was 2 and without 20% dropout is 2.5).
A solution that is often implemented is scaling the input during inference by (1-p).

In my model I added dropout after both dense hidden layers with 20% probability.
I may have been too agressive with dropout because the training performance was comparable to that of the validation set.
Overall the performance was better than without it.

### NN-3

For an overview I set the initial number of epochs to 1000 and monitored the validation test accuracy.
Afterwards I got a better idea and run a cross validation on a smaller sample of total number of epochs.
Ideally one would use a gridsearch because the learning rate is tightly coupled with how long one should train the model.

### NN-4

Examined hyperparameters/architecture decisions:
- optimizer: both hyperparameters had a large impact on the performance
  - learning rate
  - weight decay
- number of epochs: the validation performance varied up to one percent among epochs and hence it was difficult to estimate which one is the best one
- batch size: did not appear to have a large impact
- architecture: was important but I focused mostly on architectures with 2 or 3 layers 
  - number of hidden layers
  - number of neurons in the hidden layer: 
- activation function: from ReLU, leaky ReLU, TanH and Sigmoid, the ReLU had the best results
- dropout: without it, the model was overfitting
- loss: KL-divergence and categorical cross-entropy had basically the same results 
- label smoothing: the idea was to prevent incorrectly labeled examples have a large effect on the parameter updates but it did not help
- regularization: L1 and L2 regularization of the weights in the dense layers, no improvement

The most important parameters were related to the optimizer (type, learning rate and weight decay), the architecture (number of hidden layers, number of neurons) and number of epochs.

My search could be summarized as iterative one-dimensional gridsearch:

```
1: start with random (reasonable) hyperparameters
2: for every hyperparameter h:
3:   find optimum of h on validation set
4:   set model hyperparameter to this optimum
5: if noticeable improvement in performance occured, goto 2
```

Obviously this algorithm is far from perfect and can easily converge to a local maximum.
At the end this was harder to evaluate because of the diminishing improvements in performance and noise in evaluation.
I may have also overused the 10% of the development set and overfitted the hyperparameters by making too many decisions based on the reported results.

### NN-5

I started with the baseline model and changed the optimizer to Adam with weight decay of (0.01).
This single change led to the biggest improvement from validation accuracy of 64.85% to 76.79%.

I then performed the aforementioned hyperparameter-tuning algorithm and converged to the following configuration:
```
architecture:    L(300,48) ReLU Dropout(0.2) L(48, 48) ReLU Dropout(0.2) L(48, 5)
learning rate:   0.0015
weight_decay:    0.008
bach_size:       64
epochs:          11
loss:            categorical cross entropy
label smoothing: 0
regularization:  none
``` 

After the submission of the first model which scored 76.6% on the hidden test-set I began using 10-fold CV with 5 runs to find out whether changing the parameters leads to improvements.
I made miniscule changes (e.g. changing number of epochs from 11 to 10 or decreasing batch size) though they did not yield better results on the test set and improved on the validation set probably due to chance.

At the end I simply trained 50 models (single training on 10 epochs took ~7 seconds) and put them in a voting ensemble.

### NN-6

Using the ensemble I was able to identify cases in which the voting was not overwhelmingly in favor of a single class.
Such cases were almost always unclear even to me, a possible human annotator:

Token      | Predicted| True
-----------|----------|----
37         | CARDINAL | DATE
1000       | CARDINAL | DATE
04         | DATE     | CARDINAL
deby       | PERSON   | ORG
woolworth  | PERSON   | ORG
stone      | PERSON   | ORG
literacy   | ORG      | PERSON
catholicism| ORG      | GPE
medina     | PERSON   | ORG
mar        | ORG      | GPE
changzhou  | GPE      | ORG

Given the training data, I think that it's not possible to increase the performance further by a large margin.
Specifically samples with numbers seem questionable when taken out of context and e.g. the classification of 37 as a date seems like a mistake in annotation (see rationale for label smoothing in NN-4).
I find the incorrect prediction of `woolfsworth` suprising. The surface form could serve as a last name but this is not utilized in Glove and one would expect that the word would occur in places where organizations/other retailer chains occur.

### WE-1

I first examined the homonym `address` to see if it's closer to the meaning of a location or of speaking to someone. I expected the former because of the frequency. The output begins my morphological derivates of the word but then follows a list of email addresses.
In fact, out of top 500 closes words (I modified `distance.c`), 229 are email addresses. The explanation is clear: the email address usually follows after the word `address`.

Word | Cosine Distance
|-|-|
addresses	|	0.741276
addressing	|	0.650099
addressed	|	0.648243
Address	|	0.641312
addresss	|	0.602537
adress	|	0.579881
addres	|	0.547298
Email_Brummett	|	0.527992
Joab_e_mail	|	0.517807
solve	|	0.503041
Joab_Jackson@idg .com	|	0.490110
kgray@unionleader .com	|	0.476449
blewis@dentonrc .com	|	0.467501
jrutter@lnpnews .com	|	0.467155


The word `well`, also being a homonym, has the following closest words. 
Even though the words do not all represent the same positive sentiment (e.g. poorly), they occur in similar contexts.

Word | Cosine Distance
-|-
as	|	0.614120
poorly	|	0.558148
nicely	|	0.548434
excellently	|	0.515061
decently	|	0.497770
good	|	0.477837
reasonably	|	0.468224
far	|	0.451972
much	|	0.445906
such	|	0.444710


Interestingly, the words related to `rose` only concern the movement and not the plant:

Word | Cosine Distance
-|-
surged	|	0.805620
climbed	|	0.804511
soared	|	0.769516
fell	|	0.768852
tumbled	|	0.725606
dipped	|	0.720985
jumped	|	0.699768
inched	|	0.679103
risen	|	0.668675

In contrast to this, the homonym `stalk` has both words related to the verb and the part of the plant in its vicinity:

Word | Cosine Distance
-|-
stalks	|	0.680984
stalking	|	0.518083
sawfly_larvae	|	0.516364
hornworms	|	0.511791
tomato_hornworms	|	0.509612
stalked	|	0.506061
vine_weevil	|	0.493138
stalkers	|	0.476341
sap_sucking	|	0.471879
hornworm	|	0.469148

Lastly, looking at the word `first` we can notice that the similarity follows quite a good numberical ordering, up to the word `ten` (other words filtered out):

Word | Cosine Distance
-|-
second	|	0.797189
third	|	0.693208
fourth	|	0.673237
fifth	|	0.657148
sixth	|	0.623786
seventh	|	0.591549
eighth	|	0.555811
ninth	|	0.545992
eleventh	|	0.538080
thirteenth	|	0.488215
fourteenth	|	0.474172
twelfth	|	0.473737
tenth	|	0.460698


### WE-2

In lots of slavic languages there is a male and a female version of a name. This allows for using the analogy `{MALE_NAME} - man + woman` to generate a female version of the name. This actually works with Czech word embeddings. Unfortunately not with these ones (still generates a name of popular mostly female words). In the case of `Daniel`, the name `Danielle` is unfortunately not on the list, possibly because of low-frequency occurence

Word | Distance
-|-|-
Daniel - man + woman | Christine	|	0.651875
| | Rebecca	|	0.648870
| | Jennifer	|	0.625211
| | Matthew	|	0.621487
| | Leah	|	0.605731
| | |
Anthony - man + woman | Angela	|	0.592559
| | Joseph	|	0.583022
| | Dominic	|	0.573644
| | Rosanna	|	0.572420
| | Catherine	|	0.567700
| | |
| William - man + woman | Edward	|	0.602360
| | Margaret	|	0.600078
| | Catherine	|	0.595653
| | Mary	|	0.594282
| | Pamela	|	0.591181


I also tried to find out who lives in the pond. It turns out it's geese and snapping turtles.

Analogy | Word | Distance
-|-|-
deer - forest + pond | geese	|	0.522082
| | ponds	|	0.516901
| | retention_pond	|	0.492275
| | livewell	|	0.491924
| | polliwogs	|	0.487612
| | snapping_turtles	|	0.479871
| | snapping_turtle	|	0.477647
| | |
forest - deer + sky | geese	|	0.469676
| | antlered_buck	|	0.457223
| | mourning_doves	|	0.454269
| | |
| | forest - deer + sea | gulls	|	0.488235
| | goldeneyes	|	0.462562
| | gannets	|	0.454313
| | whales	|	0.454125
| | pelagic_species	|	0.445999

Finally, it's possible (mostly) to make things better using .

Analogy | Word | Distance
-|-|-
good - bad + sad | wonderful	|	0.641493
| | happy	|	0.615434
| | great	|	0.580368
| | nice	|	0.568397
| | |
good - bad + despair | joy	|	0.573050
| | utter_despair	|	0.555167
| | hopelessness	|	0.548456
| | |
good - bad + terrible | great	|	0.765869
| | terrific	|	0.695784
| | horrible	|	0.674380
| | fantastic	|	0.660054
| | wonderful	|	0.648352

## Live competititon

|N|Architecture|Parameters|CV ACC|Test ACC|
|-|-|-|-|-|
|1|L48-relu-D0.2-L48-relu-D0.2-L48-softmax|lr 0.0015, weight decase 0.008, batch_size 64, epochs 11|76.93%|76.6%|
|2|ensemble of model 1|seq 15|76.93%|76.6%|
|3|ensemble of model 1|seq 40, epochs 10|76.93%|72.0%|
|4|ensemble of model 1|seq 20, epochs 11, fix None return|76.93%|77.0%|
|5|ensemble of model 1|seq 20, epochs 11, width 52|76.93%|76.8%|
