import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.models import load_model
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
# print(words)
# print(classes)
model = load_model('chatbotmodel.h5')

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words
# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bag_of_words(sentence):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words) 
    for w in sentence_words:
        for i,word in enumerate(words):
            if word == w: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                # if show_details:
                #     print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence):
    # filter out predictions below a threshold
    bow = bag_of_words(sentence)
    # print(bow)
    res = model.predict(np.array([bow]))[0]
    # print(res)
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # print(results)
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    # print(results)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    # print(return_list)
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    # print(tag)
    list_of_intents = intents_json['intents']
    # print(list_of_intents)
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result


# test_data = [
#     {"query": "How's the weather today?", "intent": "weather"},
#     {"query": "Tell me a joke", "intent": "jokes"},
#     # Add more test data as needed
# ]

# # Initialize lists to store true and predicted labels
# true_labels = []
# predicted_labels = []

# # Make predictions for the test data and collect true labels
# for example in test_data:
#     query = example["query"]
#     true_intent = example["intent"]
#     predicted_intent = predict_class(query)[0]["intent"]

#     true_labels.append(true_intent)
#     predicted_labels.append(predicted_intent)

# # Calculate and print the confusion matrix and classification report
# confusion_mtx = confusion_matrix(true_labels, predicted_labels, labels=classes)
# class_report = classification_report(true_labels, predicted_labels, labels=classes)

# print("Confusion Matrix:")
# print(confusion_mtx)

# print("\nClassification Report:")
# print(class_report)

# # Calculate accuracy from the confusion matrix
# accuracy = (np.trace(confusion_mtx) / np.sum(confusion_mtx)) * 100

# print("Accuracy:", accuracy)

print("GO! Bot is running!")

while True:
    message=input("")
    ints=predict_class(message)
    res=get_response(ints,intents)
    print(res)
