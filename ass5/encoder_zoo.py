from shutil import ExecError
import numpy as np
from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertModel
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import torch


def encoder_tfidf(X_data, max_features):
    reviews_all = TfidfVectorizer(
        max_features=max_features).fit_transform(X_data)
    print(reviews_all.shape)
    return reviews_all.toarray()


def encoder_glove(X_data, embeddings, action):
    reviews_all = []
    for review in X_data:
        zero_vec = np.zeros(embeddings["the"].shape)
        review = np.array(
            [embeddings[x] if x in embeddings else zero_vec for x in review.split()])
        if action == "avg":
            review = np.average(review, axis=0)
        elif action == "max":
            review = np.max(review, axis=0)
        else:
            raise Exception("Unknown aggregating action")
        reviews_all.append(review)
    reviews_all = np.array(reviews_all)
    return reviews_all


def mean_pooling(model_output, attention_mask, layer_i=0):
    # Mean Pooling - Take attention mask into account for correct averaging
    # first element of model_output contains all token embeddings
    token_embeddings = model_output[layer_i]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return (sum_embeddings / sum_mask).reshape(-1)


def encoder_bert(X_data, type_out):
    reviews_all = []

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    model = BertModel.from_pretrained(
        "bert-base-cased", return_dict=True, output_hidden_states=True)
    model.train(False)

    for review in X_data:
        encoded_input = tokenizer(
            review, padding=True,
            truncation=True, max_length=128,
            return_tensors='pt'
        )
        with torch.no_grad():
            output = model(**encoded_input)

        if type_out == "cls":
            review = output[0][0, 0].cpu().numpy()
        elif type_out == "pooler":
            review = output["pooler_output"][0].cpu().numpy()
        elif type_out == "tokens":
            sentence_embedding = mean_pooling(
                output, encoded_input['attention_mask']
            )
            review = sentence_embedding.cpu().numpy()
        else:
            raise Exception("Unknown type out")

        reviews_all.append(review)

    reviews_all = np.array(reviews_all)
    print(reviews_all.shape)
    return reviews_all


def encoder_sbert(X_data, type_out):
    reviews_all = []

    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/bert-base-nli-cls-token")
    model = AutoModel.from_pretrained(
        "sentence-transformers/bert-base-nli-cls-token")
    model.train(False)

    for review in X_data:
        encoded_input = tokenizer(
            review, padding=True, truncation=True, max_length=128, return_tensors='pt')
        with torch.no_grad():
            output = model(**encoded_input)

        if type_out == "cls":
            review = output[0][0, 0].cpu().numpy()
        elif type_out == "pooler":
            review = output["pooler_output"][0].cpu().numpy()
        elif type_out == "tokens":
            sentence_embedding = mean_pooling(
                output, encoded_input['attention_mask']
            )
            review = sentence_embedding.cpu().numpy()
        else:
            raise Exception("Unknown type out")

        reviews_all.append(review)

    reviews_all = np.array(reviews_all)
    print(reviews_all.shape)
    return reviews_all


def encoder_dpr(X_data, type_out, version):
    reviews_all = []

    if version == "query":
        tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            'facebook/dpr-question_encoder-single-nq-base'
        )
        model = DPRQuestionEncoder.from_pretrained(
            'facebook/dpr-question_encoder-single-nq-base'
        )
    elif version == "doc":
        tokenizer = DPRContextEncoderTokenizer.from_pretrained(
            'facebook/dpr-ctx_encoder-single-nq-base'
        )
        model = DPRContextEncoder.from_pretrained(
            'facebook/dpr-ctx_encoder-single-nq-base'
        )
    else:
        raise Exception("Unknown model version")
    model.train(False)

    for review in X_data:
        encoded_input = tokenizer(
            review, padding=True,
            truncation=True, max_length=128,
            return_tensors='pt'
        )
        with torch.no_grad():
            output = model(
                **encoded_input, output_hidden_states=type_out in {"tokens", "cls"}
            )

        if type_out == "cls":
            review = output["hidden_states"][-1][0, 0].cpu().numpy()
        elif type_out == "pooler":
            review = output.pooler_output[0].detach().cpu().numpy()
        elif type_out == "tokens":
            sentence_embedding = mean_pooling(
                output["hidden_states"], encoded_input['attention_mask'], layer_i=-1
            )
            review = sentence_embedding.cpu().numpy()
        else:
            raise Exception("Unknown type out")

        reviews_all.append(review)

    reviews_all = np.array(reviews_all)
    print(reviews_all.shape)
    return reviews_all
