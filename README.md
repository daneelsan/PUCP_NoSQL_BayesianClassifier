# [PUCP] NoSQL - Bayesian Network Classifier

## Project Structure

```
- fraud_credit_card.zip // The compressed data. Unzip this before working running upload.py
- upload.py             // Module in charge of uploading the .csv into MongoDB
- interface.ipynb       // Module in charge of presenting an interface of the classifier to the user
- classify.py           // Module in charge of classifing using Bayesian Networks and the MongoDB database
```

## Upload to MongoDB

```python
$ time python3 upload.py
```

## Interface

Simply run the `interface.ipynb` in VScode.

