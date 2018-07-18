import numpy as np
import pickle
import os
from keras.layers import Input, Conv2D, AveragePooling2D, BatchNormalization, Add, Activation, Flatten, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.datasets import cifar10
from keras.utils import to_categorical

class TestModel:
    def __init__(self, use_resblock, nb_blocks):
        self.use_resblock = use_resblock
        self.nb_blocks = nb_blocks
        # モデルの作成
        self.model = self._create_model()
        # モデル名
        self.name = ""
        if use_resblock: self.name += "use_res_"
        else: self.name += "no_res_"
        self.name = f"{self.name}{self.nb_blocks:02d}"

    def _create_model(self):
        input = Input(shape=(32, 32, 3))
        X = input
        n_filter = 16
        for i in range(self.nb_blocks):
            # 3ブロック単位でAveragePoolingを入れる、フィルター数を倍にする
            if i % 3 == 0 and i != 0:
                X = AveragePooling2D((2,2))(X)
                n_filter *= 2
            # ショートカットとメインのフィルター数を揃えるために活性化関数なしの畳込みレイヤーを作る
            if i % 3 == 0:
                X = Conv2D(n_filter, (3,3), padding="same")(X)
            # 1ブロック単位の処理
            if self.use_resblock:
                # ショートカット：ショートカット→BatchNorm（ResBlockを使う場合のみ）
                shortcut = X
                shortcut = BatchNormalization()(shortcut)
            # メイン
            # 畳み込み→BatchNorm→活性化関数
            X = Conv2D(n_filter, (3,3), padding="same")(X)
            X = BatchNormalization()(X)
            X = Activation("relu")(X)
            # 畳み込み→BatchNorm
            X = Conv2D(n_filter, (3,3), padding="same")(X)
            X = BatchNormalization()(X)
            if self.use_resblock:
                # ショートカットとマージ（ResBlockを使う場合のみ）
                X = Add()([X, shortcut])
            # 活性化関数
            X = Activation("relu")(X)
        # 全結合
        X = Flatten()(X)
        y = Dense(10, activation="softmax")(X)
        # モデル
        model = Model(inputs=input, outputs=y)
        return model

    def train(self, Xtrain, ytrain, Xval, yval, nb_epoch=100, learning_rate=0.01):
        self.model.compile(optimizer=Adam(lr=learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])
        history = self.model.fit(Xtrain, ytrain, batch_size=128, epochs=nb_epoch, validation_data=(Xval, yval)).history
        # historyの保存
        if not os.path.exists("history"): os.mkdir("history")
        with open(f"history/{self.name}.dat", "wb") as fp:
            pickle.dump(history, fp)

if __name__ == "__main__":
    model = TestModel(True, 12)
    model.model.summary()
    exit()

    # データの読み込み
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)
    # テストパターン
    resflag = [False, True]
    nb_blocks = [3, 6, 9, 12]
    # モデルの作成
    for res in resflag:
        for nb in nb_blocks:
            print("Testing model... / ", res, nb)
            model = TestModel(res, nb)
            model.train(X_train, y_train, X_test, y_test, nb_epoch=100)