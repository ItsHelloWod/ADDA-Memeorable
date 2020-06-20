from data import  *
from tensorflow import keras


class ADDA():
    def __init__(self, img_shape=(4608, 1, 1), betas=(0.5, 0.999), lr1=1e-4, epochs=10,
                 lr2=2e-4, dropout=0.2, slope=0.1):
        # variables
        self.img_shape = img_shape
        self.es = 'es'
        self.et = 'et'
        self.c = 'c'
        self.d = 'd'
        self.lr1 = lr1
        self.lr2 = lr2
        self.betas = betas
        self.epochs = epochs
        self.dropout = dropout
        self.slope = slope

    class encoder():
        from functools import partial
        def __init__(self, name, slope=0.1, dropout=0.2):
            self.name = name
            self.slope = slope
            self.dropout = dropout

        def encode(self, img_shape=(4608, 1, 1), reuse=False, trainable=True):
            from functools import partial
            with tf.compat.v1.variable_scope(self.name, reuse=reuse):
                inp = keras.Input(shape=img_shape)
                # layers
                flat = keras.layers.Flatten(name='flat')(inp)
                dp1 = keras.layers.Dropout(self.dropout, name='dp1')(flat)
                dense1 = keras.layers.Dense(1024, activation=partial(tf.nn.leaky_relu, alpha=self.slope),
                                            trainable=trainable, name='e_dense1')(dp1)
                dp2 = keras.layers.Dropout(self.dropout, name='dp2')(dense1)
                dense2 = keras.layers.Dense(512, activation=partial(tf.nn.leaky_relu, alpha=self.slope),
                                            trainable=trainable, name='e_dense2')(dp2)
                return keras.Model(inputs=(inp), outputs=(dense2))

    def discrimnator(self, model, reuse=False, trainable=True):
        from functools import partial
        with tf.compat.v1.variable_scope(self.c, reuse=reuse):
            # layers
            dense1 = keras.layers.Dense(256, trainable=trainable, activation=partial(tf.nn.leaky_relu, alpha=0.1),
                                        name='dense1')(model.output)
            dense2 = keras.layers.Dense(128, trainable=trainable, activation=partial(tf.nn.leaky_relu, alpha=0.1),
                                        name='dense2')(dense1)
            dense3 = keras.layers.Dense(64, trainable=trainable, activation=partial(tf.nn.leaky_relu, alpha=0.1),
                                        name='dense3')(dense2)
            dense4 = keras.layers.Dense(1, trainable=trainable, activation=tf.nn.sigmoid, name='dense4')(dense3)
            return keras.Model(inputs=(model.input), outputs=(dense4))

    def classifier(self, model, reuse=False, trainable=True):
        from functools import partial
        with tf.compat.v1.variable_scope(self.c, reuse=reuse):
            # layers
            dense1 = keras.layers.Dense(256, trainable=trainable, activation=partial(tf.nn.leaky_relu, alpha=0.1),
                                        name='dense1')(model.output)
            dense2 = keras.layers.Dense(64, trainable=trainable, activation=partial(tf.nn.leaky_relu, alpha=0.1),
                                        name='dense2')(dense1)
            dense3 = keras.layers.Dense(1, trainable=trainable, activation=tf.nn.sigmoid, name='dense3')(dense2)
            return keras.Model(inputs=(model.input), outputs=(dense3))

    def build_encoder_s(self):
        return self.encoder(self.es, self.slope, self.dropout)

    def build_encoder_t(self):
        return self.encoder(self.et, self.slope, self.dropout)

    def classify(self, train_generator):
        # create encoder of source domian and classifier for source domain
        encoder_es = self.build_encoder_s().encode()
        cls_s = self.classifier(encoder_es)

        cls_s.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        cls_s.fit(train_generator, epochs=self.epochs, verbose=None)

    def discriminate(self, train_generator, test_generator):
        # create encoder of target domian and discrimnator for target domain and source domain
        encoder_et = self.build_encoder_t().encode()
        encoder_es = self.build_encoder_s().encode(reuse=True, trainable=False)
        dis_t = self.discrimnator(encoder_et)
        dis_s = self.discrimnator(encoder_es, reuse=True)

        adver_loss = tf.keras.losses.BinaryCrossentropy()

        optimizer = tf.keras.optimizers.Adam()

        for epoch in range(self.epochs):
            for i, Xt in enumerate(test_generator):
                for Xs, _ in train_generator:
                    break;

                if i > 20:
                    break
                # domain label 1 for source, 0 for target
                valid = tf.ones_like(Xs.shape[0], 1)
                fake = tf.zeros_like(Xs.shape[0], 1)
                with tf.GradientTape(persistent=True) as tape:
                    loss1 = adver_loss(valid, dis_t(Xt))
                    loss2 = adver_loss(fake, dis_t(Xt)) + adver_loss(valid, dis_s(Xs))
                gradients1 = tape.gradient(loss1, dis_t.trainable_variables)
                gradients2 = tape.gradient(loss2, dis_s.trainable_variables)
                optimizer.apply_gradients(zip(gradients1, dis_t.trainable_variables))
                optimizer.apply_gradients(zip(gradients2, dis_s.trainable_variables))

    def predict(self, test_generator):
        encoder_et = self.build_encoder_t().encode(reuse=True, trainable=False)
        classifier_t = self.classifier(encoder_et, reuse=True, trainable=False)
        predictions = []

        for i, X in enumerate(test_generator):
            prediction = classifier_t(X).numpy().reshape(-1)
            if i % 10 == 0:
                print(i)
            if predictions is None:
                predictions = prediction
            else:
                predictions = np.r_[predictions, prediction]

        predictions = (predictions > 0.5).astype(np.int32)
        return predictions