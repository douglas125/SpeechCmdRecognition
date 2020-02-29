from tensorflow.keras.models import Model, load_model

from tensorflow.keras import layers as L
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers

from kapre.time_frequency import Melspectrogram, Spectrogram
from kapre.utils import Normalization2D


def ConvSpeechModel(nCategories, samplingrate=16000, inputLength=16000):
    """
    Base fully convolutional model for speech recognition
    """

    inputs = L.Input((inputLength,))

    x = L.Reshape((1, -1))(inputs)

    x = Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1, inputLength),
                       padding='same', sr=samplingrate, n_mels=80,
                       fmin=40.0, fmax=samplingrate / 2, power_melgram=1.0,
                       return_decibel_melgram=True, trainable_fb=False,
                       trainable_kernel=False,
                       name='mel_stft')(x)

    x = Normalization2D(int_axis=0)(x)
    # note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
    # we would rather have it the other way around for LSTMs

    x = L.Permute((2, 1, 3))(x)
    # x = Reshape((94,80)) (x) #this is strange - but now we have (batch_size,
    # sequence, vec_dim)

    c1 = L.Conv2D(20, (5, 1), activation='relu', padding='same')(x)
    c1 = L.BatchNormalization()(c1)
    p1 = L.MaxPooling2D((2, 1))(c1)
    p1 = L.Dropout(0.03)(p1)

    c2 = L.Conv2D(40, (3, 3), activation='relu', padding='same')(p1)
    c2 = L.BatchNormalization()(c2)
    p2 = L.MaxPooling2D((2, 2))(c2)
    p2 = L.Dropout(0.01)(p2)

    c3 = L.Conv2D(80, (3, 3), activation='relu', padding='same')(p2)
    c3 = L.BatchNormalization()(c3)
    p3 = L.MaxPooling2D((2, 2))(c3)

    p3 = L.Flatten()(p3)
    p3 = L.Dense(64, activation='relu')(p3)
    p3 = L.Dense(32, activation='relu')(p3)

    output = L.Dense(nCategories, activation='softmax')(p3)

    model = Model(inputs=[inputs], outputs=[output], name='ConvSpeechModel')

    return model


def RNNSpeechModel(nCategories, samplingrate=16000, inputLength=16000):
    # simple LSTM
    sr = samplingrate
    iLen = inputLength

    inputs = L.L.Input((iLen,))

    x = L.Reshape((1, -1))(inputs)

    x = Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1, iLen),
                       padding='same', sr=sr, n_mels=80,
                       fmin=40.0, fmax=sr / 2, power_melgram=1.0,
                       return_decibel_melgram=True, trainable_fb=False,
                       trainable_kernel=False,
                       name='mel_stft')(x)

    x = Normalization2D(int_axis=0)(x)

    # note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
    # we would rather have it the other way around for LSTMs

    x = L.Permute((2, 1, 3))(x)

    x = L.Conv2D(10, (5, 1), activation='relu', padding='same')(x)
    x = L.BatchNormalization()(x)
    x = L.Conv2D(1, (5, 1), activation='relu', padding='same')(x)
    x = L.BatchNormalization()(x)

    # x = Reshape((125, 80)) (x)
    # keras.backend.squeeze(x, axis)
    x = L.Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim')(x)

    x = L.Bidirectional(L.CuDNNLSTM(64, return_sequences=True))(
        x)  # [b_s, seq_len, vec_dim]
    x = L.Bidirectional(L.CuDNNLSTM(64))(x)

    x = L.Dense(64, activation='relu')(x)
    x = L.Dense(32, activation='relu')(x)

    output = L.Dense(nCategories, activation='softmax')(x)

    model = Model(inputs=[inputs], outputs=[output])

    return model


def AttRNNSpeechModel(nCategories, samplingrate=16000,
                      inputLength=16000, rnn_func=L.LSTM):
    # simple LSTM
    sr = samplingrate
    iLen = inputLength

    inputs = L.Input((inputLength,), name='input')

    x = L.Reshape((1, -1))(inputs)

    m = Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1, iLen),
                       padding='same', sr=sr, n_mels=80,
                       fmin=40.0, fmax=sr / 2, power_melgram=1.0,
                       return_decibel_melgram=True, trainable_fb=False,
                       trainable_kernel=False,
                       name='mel_stft')
    m.trainable = False

    x = m(x)

    x = Normalization2D(int_axis=0, name='mel_stft_norm')(x)

    # note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
    # we would rather have it the other way around for LSTMs

    x = L.Permute((2, 1, 3))(x)

    x = L.Conv2D(10, (5, 1), activation='relu', padding='same')(x)
    x = L.BatchNormalization()(x)
    x = L.Conv2D(1, (5, 1), activation='relu', padding='same')(x)
    x = L.BatchNormalization()(x)

    # x = Reshape((125, 80)) (x)
    # keras.backend.squeeze(x, axis)
    x = L.Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim')(x)

    x = L.Bidirectional(rnn_func(64, return_sequences=True)
                        )(x)  # [b_s, seq_len, vec_dim]
    x = L.Bidirectional(rnn_func(64, return_sequences=True)
                        )(x)  # [b_s, seq_len, vec_dim]

    xFirst = L.Lambda(lambda q: q[:, -1])(x)  # [b_s, vec_dim]
    query = L.Dense(128)(xFirst)

    # dot product attention
    attScores = L.Dot(axes=[1, 2])([query, x])
    attScores = L.Softmax(name='attSoftmax')(attScores)  # [b_s, seq_len]

    # rescale sequence
    attVector = L.Dot(axes=[1, 1])([attScores, x])  # [b_s, vec_dim]

    x = L.Dense(64, activation='relu')(attVector)
    x = L.Dense(32)(x)

    output = L.Dense(nCategories, activation='softmax', name='output')(x)

    model = Model(inputs=[inputs], outputs=[output])

    return model
