import numpy as np
import tensorflow as tf
import random

def trainSnn(model, epoch_num, train_data):
    y_dummie = np.ndarray([1,1,1])

    for epoch in range(epoch_num):
        print("epoch: " + str(epoch + 1))
        for triplet in train_data:
            model.fit(x=triplet, y=y_dummie)

    return model


def testSnn(model, test_data):
    correct_preds = 0
    for i in range(len(test_data)):
        encodings = model.predict(test_data[i])
        anchor = tf.convert_to_tensor(encodings[0])
        positive = tf.convert_to_tensor(encodings[1]) 
        negative = tf.convert_to_tensor(encodings[2])
        
        cosine_similarity = tf.metrics.CosineSimilarity()

        positive_similarity = cosine_similarity(anchor, positive)
        print("Positive similarity:", positive_similarity.numpy())

        cosine_similarity = tf.metrics.CosineSimilarity()

        negative_similarity = cosine_similarity(anchor, negative)
        print("Negative similarity", negative_similarity.numpy())

        if (positive_similarity.numpy() > negative_similarity.numpy()):
            correct_preds = correct_preds + 1
            print("Correct prediction!")

    print("Correct predications:", correct_preds)
    print("Accuracy", (correct_preds/len(test_data))*100,'%')

    return (correct_preds/len(test_data))*100

