import matplotlib
import numpy
from keras.layers import LSTM, Dense
from keras.metrics import mean_squared_error
from keras.models import Sequential
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

data = genfromtxt('data/layton_wild', delimiter=',')

data = data[1:]

scaler = MinMaxScaler(feature_range=(-1, 1))

emg = data[::2, 1:2]
emg = scaler.fit_transform(emg)

acc = data[1::2, 1:-1]
acc = scaler.fit_transform(acc)


# plt.plot(emg)
# plt.show()


def recurrent_model():
    # Epoch 30000/30000
    # 0s - loss: 0.6794 - acc: 0.5000

    features = 4
    output_dim = 3
    look_back = 25
    testcases = min(emg.shape[0], acc.shape[0]) - look_back
    X_rec = numpy.ndarray(shape=(testcases, look_back + 1, features), dtype=float)
    Y_rec = numpy.ndarray(shape=(testcases, output_dim), dtype=float)

    for i in range(testcases):
        for j in range(look_back + 1):
            X_rec[i, j, :] = numpy.append(acc[j + i], emg[j + i])
        Y_rec[i] = acc[i + look_back][:]

    # train_samples = int(testcases * 0.66)
    # X_rec_train = X_rec[:train_samples, :, :]
    # X_rec_test = X_rec[train_samples:, :, :]
    # Y_rec_train = Y_rec[:train_samples]
    # Y_rec_test = Y_rec[train_samples:]

    model = Sequential()
    model.add(LSTM(output_dim, input_dim=features))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
    # Fit the model
    model.fit(X_rec, Y_rec, validation_split=0.3, nb_epoch=4, batch_size=10, verbose=2)

    # make predictions
    predict = model.predict(X_rec)
    # invert predictions
    # predict = scaler.inverse_transform(predict)
    # Y = scaler.inverse_transform([Y_rec])
    # predict = scaler.inverse_transform(predict)
    # Y = scaler.inverse_transform([Y])
    # calculate root mean squared error
    # trainScore = numpy.sqrt(mean_squared_error(Y_rec[0], predict[:, 0]))
    # print('Train Score: %.2f RMSE' % (trainScore))

    # shift train predictions for plotting
    target = Y_rec[:, 2:3]
    zPredict = predict[:, 2:3]

    trainPredictPlot = numpy.empty_like(acc)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(zPredict) + look_back, :] = zPredict

    # plot baseline and predictions
    plt.plot(target, label='actual')
    plt.plot(trainPredictPlot, label='predict')

    plt.savefig('graph.png')

    plt.show()


recurrent_model()

print('Done.')
