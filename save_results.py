from data import *
from adda import *

if __name__ == '__main__':
    adda = ADDA()

    adda.classify(train_generator)
    adda.discriminate(train_generator, test_generator)
    predictions = adda.predict(test_generator)

    predictions = list(zip(1, range(predictions.shape[0] + 1), predictions))
    predictions = pd.DataFrame(data=predictions, columns=['ID', 'prediction'])
    predictions.to_csv("submission_ADDA.csv", index=False, header=True)