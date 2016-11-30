from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.optimizers import SGD, Adagrad
from sklearn.metrics import log_loss
from keras import __version__ as keras_version
from keras.callbacks import EarlyStopping
from sklearn.cross_validation import KFold
import utils


def create_model():
    """
    Create the CNN structure
    :return:
    """
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 48, 48), dim_ordering='th'))
    model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='th', init='he_uniform'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='th', init='he_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))

    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th', init='he_uniform'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th', init='he_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))

    model.add(Flatten())
    model.add(Dense(96, activation='relu', init='he_uniform'))
    model.add(Dropout(0.515))
    model.add(Dense(16, activation='relu', init='he_uniform'))
    model.add(Dropout(0.515))
    model.add(Dense(8, activation='softmax'))

    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.898, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    return model


def run_cross_validation_create_models(nfolds=10):
    # input image dimensions
    batch_size = 24
    nb_epoch = 15
    random_state = 51
    first_rl = 96

    train_data, train_target, train_id = utils.read_and_normalize_train_data()

    yfull_train = dict()
    kf = KFold(len(train_id), n_folds=nfolds, shuffle=True, random_state=random_state)
    num_fold = 0
    sum_score = 0
    models = []
    for train_index, test_index in kf:
        model = create_model()
        X_train = train_data[train_index]
        Y_train = train_target[train_index]
        X_valid = train_data[test_index]
        Y_valid = train_target[test_index]

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, verbose=0),
        ]
        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                  shuffle=True, verbose=2, validation_data=(X_valid, Y_valid),
                  callbacks=callbacks)

        predictions_valid = model.predict(X_valid.astype('float32'), batch_size=batch_size, verbose=2)
        score = log_loss(Y_valid, predictions_valid)
        print('Score log_loss: ', score)
        sum_score += score * len(test_index)

        # Store valid predictions
        for i in range(len(test_index)):
            yfull_train[test_index[i]] = predictions_valid[i]

        models.append(model)

    score = sum_score / len(train_data)
    print("Log_loss train independent avg: ", score)

    info_string = 'loss_' + str(score) + '_folds_' + str(nfolds) + '_ep_' + str(nb_epoch) + '_fl_' + str(first_rl)
    return info_string, models


def run_cross_validation_process_test(info_string, models):
    batch_size = 24
    num_fold = 0
    yfull_test = []
    test_id = []
    nfolds = len(models)

    for i in range(nfolds):
        model = models[i]
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        test_data, test_id = utils.read_and_normalize_test_data()
        test_prediction = model.predict(test_data, batch_size=batch_size, verbose=2)
        yfull_test.append(test_prediction)

    test_res = utils.merge_several_folds_mean(yfull_test, nfolds)
    info_string = 'loss_' + info_string \
                  + '_folds_' + str(nfolds)
    utils.create_submission(test_res, test_id, info_string)


if __name__ == '__main__':
    print('Keras version: {}'.format(keras_version))
    num_folds = 3
    info_string, models = run_cross_validation_create_models(num_folds)
    run_cross_validation_process_test(info_string, models)

