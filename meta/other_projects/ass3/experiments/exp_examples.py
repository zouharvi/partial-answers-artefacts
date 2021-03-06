from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import sklearn.svm

def color_example(review, vocab, coefs):
    # color a message (LaTeX style) based on the coefficients
    message = ""
    for token in review:
        if token not in vocab:
            message += token + " "
        else:
            index = vocab[token]
            coef = coefs[index]
            if coef < -0.1:
                message += "\\textcolor{DarkRed}{" + token + "} "
            elif coef > 0.1:
                message += "\\textcolor{DarkGreen}{" + token + "} "
            else:
                message += token + " "
    return message


def experiment_examples(X_full, Y_full, tf_idf, max_features):
    """
    Generate a few examples with color, examine noise coefficients and craft an adversial example
    that changes the polarity of the classification.
    """

    # define and fit the model on all data and make predictions
    model = Pipeline([
        ("vec",
         TfidfVectorizer(
             preprocessor=lambda x: x,
             tokenizer=lambda x:x, max_features=max_features,
         )
         if tf_idf else
         CountVectorizer(
             preprocessor=lambda x: x,
             tokenizer=lambda x:x, max_features=max_features,
         ),
         ),
        ("svm", sklearn.svm.SVC(kernel="linear")),
    ])
    model.fit(X_full, Y_full)
    Y_pred = model.predict(X_full)

    # retrieve the vectorizer vocabulary and the model coefficients
    vocab = model.get_params()["vec"].vocabulary_
    coefs_original = model.get_params()["svm"].coef_.toarray().reshape(-1)

    # color the first 10 examples of reviews which are short (up to 100 tokens)
    for review, y_true, y_pred in [(x, y, z) for x, y, z in zip(X_full, Y_full, Y_pred) if len(x) <= 100][:10]:
        message = color_example(review, vocab, coefs_original)
        print(
            message, (y_true, y_pred),
            model.decision_function([review]), "\n"
        )

    # examine the coefficients of small tokens
    small_vocab = [k for k in vocab.keys() if len(k) <= 3]
    print("noise token   coefficient")
    for noise_token in small_vocab:
        if noise_token in vocab and all([not x.isalpha() and not x.isnumeric() for x in noise_token]):
            print(
                f"{noise_token:>11}    {coefs_original[vocab[noise_token]]:.3f}")

    # start with the following review (in top 10 in the previous part)
    # and add the noise token until it flips the prediction
    adversial = "this camera works well , except that the shutter speed is a bit slow . the image quality is decent . the use of aa rechargeable batteries is also convenient . the camera is pretty sturdy . i 've dropped it a few times and it still works fine".split()
    noise_token = "#"
    original_pred = model.predict([adversial])
    hit = False
    for i in range(50):
        adversial_tmp = adversial + [noise_token] * i
        current_pred = model.predict([adversial_tmp])
        print(i, model.predict([adversial_tmp]),
              model.decision_function([adversial_tmp]))
        if current_pred != original_pred and not hit:
            print(color_example(adversial_tmp, vocab, coefs_original))
            hit = True
