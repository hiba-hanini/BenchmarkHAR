from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    LSTM,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GaussianNoise,
    GlobalAveragePooling2D,
    GlobalMaxPooling1D,
    Input,
    LeakyReLU,
    AveragePooling2D,
    MaxPooling2D,
    SimpleRNN,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Reshape
import numpy as np

from sklearn.preprocessing import StandardScaler



def build_cnn(input_shape, n_classes):
    i = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), strides=2, activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(0.0005))(i)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 1))(x)
    x = Dropout(0.2)(x)
    x = Conv2D(64, (3, 3), strides=2, activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(0.0005))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Conv2D(128, (3, 3), strides=2, activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(0.0005))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 1))(x)
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(n_classes, activation='softmax')(x)
    model = Model(i, x)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
 
 
def build_dnn(input_shape, n_classes):
    i = Input(shape=input_shape)
    x = Flatten()(i)

    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(n_classes, activation='softmax')(x)
    model = Model(i, x)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
 
 
def build_rnn(input_shape, n_classes):
    """
    Recurrent architecture with temporal pooling (LSTM-GMP).
    Applies a sequence-returning LSTM layer (256 units), followed by a dense
    projection (128 units, ReLU) and Global Max-Pooling over the temporal
    dimension to aggregate salient activations across time steps.
    """
    i = Input(shape=input_shape)

    x = SimpleRNN(256)(i)
    x = Dropout(0.2)(x)

    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    x = Dense(n_classes, activation='softmax')(x)

    model = Model(i, x)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_lstm(input_shape, n_classes):
    """
    Standard sequential recurrent architecture (LSTM-LastState).
    Relying on the final hidden state of a 128-unit standard LSTM layer,
    followed by dropout regularization (0.2) and fully connected layers.
    """
    i = Input(shape=input_shape)

    x = LSTM(128)(i)
    x = Dropout(0.2)(x)

    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(n_classes, activation='softmax')(x)

    model = Model(i, x)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def build_deepconv_lstm(input_shape, n_classes):
    i = Input(shape=input_shape)

    x = Conv2D(
        64,
        kernel_size=(3, 3),
        padding='same',
        activation='relu',
        kernel_regularizer=regularizers.l2(0.0005)
    )(i)
    x = BatchNormalization()(x)

    x = Conv2D(
        64,
        kernel_size=(3, 3),
        padding='same',
        activation='relu',
        kernel_regularizer=regularizers.l2(0.0005)
    )(x)
    x = BatchNormalization()(x)

    x = Conv2D(
        64,
        kernel_size=(3, 3),
        padding='same',
        activation='relu',
        kernel_regularizer=regularizers.l2(0.0005)
    )(x)
    x = BatchNormalization()(x)

    x = Conv2D(
        64,
        kernel_size=(3, 3),
        padding='same',
        activation='relu',
        kernel_regularizer=regularizers.l2(0.0005)
    )(x)
    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)

    # (batch, 128, 9, 64) -> (batch, 128, 9*64)
    # x = Reshape((input_shape[0], input_shape[1] * 64))(x)
    x = Reshape((input_shape[0], -1))(x)

    x = LSTM(
        128,
        kernel_regularizer=regularizers.l2(0.0005),
        recurrent_regularizer=regularizers.l2(0.0005)
    )(x)

    x = Dropout(0.5)(x)

    x = Dense(
        32,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.0005)
    )(x)

    x = Dense(
        n_classes,
        activation='softmax'
    )(x)

    model = Model(i, x)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_astar_har(input_shape, n_classes):
    """
    Re-implementation of the har_conv2d model from A-STAR-HAR project in Keras.
    Original paper: "A*HAR: A NEW BENCHMARK TOWARDS SEMI-SUPERVISED LEARNING FOR CLASS IMBALANCED HUMAN ACTIVITY RECOGNITION"
    Original implementation: https://github.com/LOKI-LOU/A-STAR-HAR
    """
    i = Input(shape=input_shape)

    # The original model has input_noise, which we can replicate with GaussianNoise.
    # The value 0.15 is from arguments.py in ASTAR-HAR
    x = GaussianNoise(0.15)(i)

    # Block 1
    x = Conv2D(128, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.1)(x)
    x = Conv2D(128, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.1)(x)
    x = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(x)
    x = Dropout(0.5)(x)

    # Block 2
    x = Conv2D(256, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.1)(x)
    x = Conv2D(256, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.1)(x)
    x = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(x)
    x = Dropout(0.5)(x)

    # Block 3
    x = Conv2D(256, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.1)(x)
    x = Conv2D(256, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.1)(x)
    x = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(x)
    x = Dropout(0.5)(x)

    # Block 4
    # The original A-STAR-HAR uses 'valid' padding, but this fails when the
    # number of input features is small (e.g., 16 for wisdm), as the width of
    # the tensor becomes < 3. Using 'same' padding is a robust fix.
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.1)(x)
    x = Conv2D(256, kernel_size=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.1)(x)
    x = Conv2D(128, kernel_size=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.1)(x)
    x = GlobalAveragePooling2D()(x)

    x = Dense(n_classes, activation='softmax')(x)

    model = Model(i, x)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


MODEL_BUILDERS = {
    'cnn': build_cnn,
    'dnn': build_dnn,
    'rnn': build_rnn,
    'lstm': build_lstm,
    'deepconv_lstm': build_deepconv_lstm,
    'astar_har': build_astar_har
}

def prepare_input(X, model_name):
    if model_name in ['cnn', 'deepconv_lstm', 'astar_har']:
        return np.expand_dims(X, -1)
    if model_name == "tinyhar":
        return X[:, None, None, :, :]
    return X

def standardize_data(model_name, X_train, X_test):
    scaler = StandardScaler()
    
    if model_name == "dnn":
        x_train = scaler.fit_transform(X_train)
        x_test = scaler.transform(X_test)

    elif model_name in ["cnn", "deepconv_lstm", "astar_har"]:
        n_train, T, F, Ch = X_train.shape
        n_test = X_test.shape[0]

        x_train = X_train.reshape(-1, F)
        x_test = X_test.reshape(-1, F)

        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        x_train = x_train.reshape(n_train, T, F, Ch)
        x_test = x_test.reshape(n_test, T, F, Ch)

    elif model_name == "tinyhar":

        n_train, C1, C2, T, F = X_train.shape
        n_test = X_test.shape[0]

        x_train = X_train.reshape(-1, F)
        x_test = X_test.reshape(-1, F)

        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        x_train = x_train.reshape(
            n_train, C1, C2, T, F
        )

        x_test = x_test.reshape(
            n_test, C1, C2, T, F
        )

    else:
        n_train, T, F = X_train.shape
        n_test = X_test.shape[0]

        x_train = X_train.reshape(-1, F)
        x_test = X_test.reshape(-1, F)

        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        x_train = x_train.reshape(n_train, T, F)
        x_test = x_test.reshape(n_test, T, F)

    return x_train, x_test, scaler


def apply_standardize(model_name, X, scaler):
    if model_name == "dnn":
        return scaler.transform(X)

    elif model_name in ["cnn", "deepconv_lstm", "astar_har"]:
        n, T, F, Ch = X.shape

        X_flat = X.reshape(-1, F)
        X_flat = scaler.transform(X_flat)

        return X_flat.reshape(n, T, F, Ch)

    elif model_name == "tinyhar":
        n, C1, C2, T, F = X.shape

        X_flat = X.reshape(-1, F)
        X_flat = scaler.transform(X_flat)

        return X_flat.reshape(
            n, C1, C2, T, F
        )

    else:
        n, T, F = X.shape

        X_flat = X.reshape(-1, F)
        X_flat = scaler.transform(X_flat)

        return X_flat.reshape(n, T, F)