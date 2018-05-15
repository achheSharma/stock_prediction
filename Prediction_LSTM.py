''' Run Using Python for 64-bit '''

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten
from sklearn.preprocessing import MinMaxScaler

def create_dl_model(x_train):
    model = Sequential()

    # Adding the first LSTM layer
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))

    # Adding a second LSTM layer
    model.add(LSTM(units = 50, return_sequences = True))
    
    # Adding a third LSTM layer
    model.add(LSTM(units = 50, return_sequences = True))

    # Adding a fourth LSTM layer
    model.add(LSTM(units = 50))

    # Adding the output layer
    model.add(Dense(units = 1))
    return model

def compile_and_run(model, x_train, y_train, epochs=50, batch_size=32):
    model.compile(metrics=['accuracy'], optimizer='adam', loss='mean_squared_error')
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=3)
    return history

def plot_metrics(history):
    metrics_df = pd.DataFrame(data={"loss": history.history['loss']})
    metrics_df.plot()
    # plt.show()

def make_predictions(x_test, y_test, scalery, model):
    y_pred = model.predict(x_test)
    final_predictions = scalery.inverse_transform(y_pred)
    fp = np.ndarray.flatten(final_predictions)

    y_test = scalery.inverse_transform(y_test)
    ap = np.ndarray.flatten(y_test)

    pdf = pd.DataFrame(data={'Actual': ap, 'Predicted': fp})
    ax = pdf.plot()

    plt.show()
    return pdf

def accuracy(result):
    tol = 0.05
    result['Result'] = np.nan
    for i in range(len(result)):
        if abs(result['Actual'].loc[i] - result['Predicted'].loc[i]) < tol*result['Actual'].loc[i]:
            result['Result'].loc[i] = 1
        else:
            result['Result'].loc[i] = 0
    # print(result)
    print(sum(result['Result'])*100/len(result['Result']))

''' File Selection '''
path = "C:\\Users\\aksha\\Desktop\\ML Hackathon\\complete_data_set_v1\\"

filename = "PCA"
df = pd.read_csv(path + filename + ".csv")

''' Data Selection & Normalisation'''
features = df.columns[0:-1]
print(features)
feature_set = df[features].values
x = np.array(feature_set)
# x = x.reshape(-1,1) #if only 1 feature is selected
scalerx = MinMaxScaler()
x = scalerx.fit_transform(x)

target_set = df['Target'].values
y = np.array(target_set)
y = y.reshape(-1,1)
scalery = MinMaxScaler()
y = scalery.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle = False)

''' Data Transformation '''
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

''' Deep Neural Net LSTM Model '''
dl_model = create_dl_model(x_train)
history = compile_and_run(dl_model, x_train, y_train, epochs=20)
# plot_metrics(history)
result = make_predictions(x_test, y_test, scalery, dl_model)
accuracy(result)

''' Predict Return for last entry of File used to build Model'''
path = ".\\Dataset\\"
training_file = "Company.NS_Features" #Select Recpective Company for Features
df = pd.read_csv(path + training_file + ".csv")
df.Date = pd.to_datetime(df.Date)
test_close = df.Close.loc[len(df)-1]

pred_return = (result['Predicted'].loc[len(result)-1]/test_close)-1

print(pred_return)