import pandas as pd
import yfinance as yf
from datetime import datetime
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

start=datetime(2019,1,1)
end=datetime.today().date()
# end.date()

st.title("Stock Prediction using Deep Learning")
user_input=st.text_input("Enter the stock Ticker","AAPL")
df=yf.download(user_input,start,end)
# df=df.reset_index().drop(["Adj Close","Date"],axis=1)


st.subheader("Data From 2019 till NOW")
st.write(df.describe())


ma100=df.Close.rolling(100).mean()
st.subheader("Closing vs Time charts with 100MA")
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,"g")
plt.plot(df.Close)
st.pyplot(fig)


ma200=df.Close.rolling(200).mean()
st.subheader("Closing vs Time charts with 200MA")
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,"g")
plt.plot(ma200,"r")

plt.plot(df.Close,"b")
st.pyplot(fig)



i=int(len(df)*0.7)
data_training=pd.DataFrame(df.Close[0:i])
data_test=pd.DataFrame(df.Close[i:])

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
data_training_array=scaler.fit_transform(data_training) 



x_train=[]
y_train=[]
import numpy as np
for i in range(100,data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])
    
    
my_x_train=np.array(x_train)
my_y_train=np.array(y_train)

model=load_model("./keras_model.h5")

past_100=data_training.tail(100)
# final_df=past_100.append(data_test,ignore_index=True)
final_df=pd.concat([past_100,data_test],ignore_index=True)
input_data=scaler.fit_transform(final_df)


x_test=[]
y_test=[]
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test=np.array(x_test)
y_test=np.array(y_test)


y_predict=model.predict(x_test)

scale_factor=1 / 0.01754694
y_predict=scale_factor*y_predict
y_test=scale_factor* y_test
# y_predict=y_predict
st.subheader("Prediction vs actual")
fig2=plt.figure(figsize=(12,6))



past_100 = final_df.tail(100)
input_data = scaler.transform(past_100)
future_predictions = []
for i in range(100):
    x_input = input_data[-100:].reshape((1, 100, 1))
    y_pred = model.predict(x_input)
    future_predictions.append(y_pred[0, 0])
    input_data = np.append(input_data, y_pred)
    input_data = np.delete(input_data, 0)
future_predictions = np.array(future_predictions)
future_predictions = future_predictions * scale_factor


plt.plot(y_test,"g",label="Original")
plt.plot(range(-10,-10+len(y_predict)),y_predict,"b",label="Predicted")

plt.plot(range(len(y_test), len(y_test) + len(future_predictions)), future_predictions, "r", label="Predict")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()
st.pyplot(fig2)
