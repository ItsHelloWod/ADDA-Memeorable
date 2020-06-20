import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import tensorflow as tf


def load_data():
    train_set1 = pd.read_csv('training.csv', index_col='ID')
    train_set2 = pd.read_csv('additional_training.csv', index_col='ID')
    label_confidence = pd.read_csv('annotation_confidence.csv', index_col='ID').values
    source_set = pd.concat((train_set1, train_set2), axis=0)

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_source = imputer.fit_transform(source_set.iloc[:, :-1])
    y_source = source_set.iloc[:, -1].values.astype(np.float32)

    # Soften the label by using the label confidence
    for i, y in enumerate(y_source):
        y_source[i] = label_confidence[i] * y_source[i] + (1 - label_confidence[i]) * (1 - y_source[i])

    # normalize the features
    train_mean = np.mean(X_source, axis=0)
    train_std = np.std(X_source, axis=0)
    X_source = (X_source - train_mean) / train_std
    X_source = tf.convert_to_tensor(X_source, dtype=tf.float32)
    y_source = tf.reshape(tf.convert_to_tensor(y_source, dtype=tf.float32), (-1, 1))

    X_target = pd.read_csv('testing.csv', index_col='ID').values
    m = np.mean(X_target, axis=0)
    s = np.std(X_target, axis=0)
    X_target = (X_target - m) / s
    X_target = tf.convert_to_tensor(X_target, dtype=tf.float32)

    return X_source, y_source, X_target


def generate_batches():
    #transform to 4 rank, in order to fit with ImageGenerator
    X_gen_source = tf.expand_dims(X_source,axis = 0)
    X_gen_source = tf.expand_dims(X_gen_source,axis = 0)
    X_gen_source = tf.reshape(X_gen_source,(2466,4608,1,1))
    X_gen_target = tf.expand_dims(X_target,axis = 0)
    X_gen_target = tf.expand_dims(X_gen_target,axis = 0)
    X_gen_target = tf.reshape(X_gen_target,(11874,4608,1,1))
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator()

    test_datagen = ImageDataGenerator()

    # Flow training images in batches of 64 using train_datagen generator
    train_generator_cls = train_datagen.flow(
            X_gen_source,
            y_source,
            batch_size=64,
            shuffle=True)
    test_generator_cls = test_datagen.flow(
            X_gen_target,
            None,
            batch_size=64,
            # Since we use binary_crossentropy loss, we need binary labels
            shuffle=True)

    return train_generator_cls,test_generator_cls

if __name__ == '__main__':
    X_source, y_source, X_target = load_data()
    train_generator,test_generator = generate_batches()