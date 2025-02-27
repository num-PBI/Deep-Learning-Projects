import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding , LSTM,Dense
from tensorflow.keras.datasets import imdb

#Load the imdb dataset into tuples
num_words=10000    #keep most frequent 10k words from all the statements
(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=num_words)

#make pad_sequences to have similar number of values for every comment.
max_len=200
x_train=pad_sequences(x_train,maxlen=max_len,padding='post')
x_test=pad_sequences(x_test,maxlen=max_len,padding='post')

#initiate model
model=Sequential([
    Embedding(input_dim=num_words,output_dim=128,input_length=max_len),
    LSTM(64,return_sequences=False),
    Dense(1,activation='sigmoid')
    
])
# fit the model on training dataset
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))
test_loss, test_acc = model.evaluate(x_test, y_test)

#Convert the user input which is a string to numbers representation from available data
def preprocess_review(review):
  words=review.lower().split()
  review_numbers=[word_index.get(word,2) for word in words]
  return pad_sequences([review_numbers],maxlen=max_len)

#take the input and let model predict the score
def predict_sentiment(review):
  processed_review=preprocess_review(review)
  prediction=model.predict(processed_review)[0][0]
  sentiment="Positive" if prediction>0.5 else " Negative"
  return sentiment,prediction
#take user input
user_review = input("Enter a movie review: ")
sentiment, confidence = predict_sentiment(user_review)
print(f"Predicted Sentiment: {sentiment} (Confidence: {confidence:.2f})")

