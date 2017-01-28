import matplotlib
import numpy
from keras.layers import LSTM, Dense
from keras.metrics import mean_squared_error
from keras.models import Sequential
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

data = genfromtxt('data/layton_wild', delimiter=',')

data = data[1:]

emg = data[::2, 1:2]
# gives data zero mean
emg = preprocessing.scale(emg)

acc = data[1::2, 1:-1]
# gives data zero mean
acc = preprocessing.scale(acc)


def plotWindow(target, predict, offset):
    """
    Shows plot in window
    """ \
        # just plot z axis for now
    target = target[:, 2:3]
    zPredict = predict[:, 2:3]

    predictPlot = numpy.empty_like(acc)
    predictPlot[:, :] = numpy.nan
    predictPlot[offset:len(zPredict) + offset, :] = zPredict

    # plot baseline and predictions
    plt.plot(target, label='actual')
    plt.plot(predictPlot, label='predict')

    plt.show()


def plotFile(target, predict, offset):
    """
    Writes plot to image file. Just showing a small part of the x-axis so it's big enough to see.
    """

    cutoffFrac = 0.9
    cutoffNum = int(cutoffFrac * len(target))

    # just plot z axis for now
    target = target[cutoffNum:, 2:3]
    zPredict = predict[cutoffNum:, 2:3]

    predictPlot = numpy.empty_like(acc)
    predictPlot[:, :] = numpy.nan
    predictPlot[offset:len(zPredict) + offset, :] = zPredict

    # plot baseline and predictions
    plt.plot(target, label='actual')
    plt.plot(predictPlot, label='predict')

    plt.savefig('graph.png')


def recurrent_model():
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

    model = Sequential()
    model.add(LSTM(output_dim, input_dim=features))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
    # Fit the model
    model.fit(X_rec, Y_rec, validation_split=0.3, nb_epoch=4, batch_size=10, verbose=2)

    # make predictions
    predict = model.predict(X_rec)

    plotFile(Y_rec, predict, look_back)
    plotWindow(Y_rec, predict, look_back)


recurrent_model()

print('Done.')
