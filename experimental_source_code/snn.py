import tensorflow as tf
import config

def create_base_network(model_path):
    """
    Base network that takes the inital input and begins learning.
    """
    if (model_path==''):
        model = tf.keras.Sequential(name="base_model")
        model.add(tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(224,224,3)))
        model.add(tf.keras.layers.MaxPooling2D(2,2))
        model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(2,2))
        model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(2,2))

        model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(2,2))
        model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(2,2))
        # model.add(ResidualBlock(64).forward(input=()))
        model.add(tf.keras.layers.Flatten())
        # model.add(tf.keras.layers.Dense(512, activation='sigmoid'))
        # model.add(tf.keras.layers.Dense(5, activation='softmax'))
    else:
        model = tf.keras.models.load_model(model_path)
    
    # trainable = False
    # for layer in model.layers:
    #     if layer.name == "dense":
    #         trainable = True
    #     layer.trainable = trainable

    return model

def lossless_triplet_loss(y_true, y_pred, N = 3, beta=3, epsilon=1e-8, margin=0.2):

    anchor = tf.convert_to_tensor(y_pred[:,0])
    positive = tf.convert_to_tensor(y_pred[:,1]) 
    negative = tf.convert_to_tensor(y_pred[:,2])
    
    # distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,positive)),-1)
    # distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,negative)),-1)
    
    #Non Linear Values  
    
    # -ln(-x/N+1)
    # pos_dist = -tf.math.log(-tf.divide((pos_dist),beta)+1+epsilon)
    # neg_dist = -tf.math.log(-tf.divide((N-neg_dist),beta)+1+epsilon)
    
    # compute loss

    loss = pos_dist - neg_dist + margin
    loss = tf.maximum(loss,0.0)
    
    return loss

def createSNN(model_path):

    model =  create_base_network(model_path)

    embedding = tf.keras.Model(inputs=model.input, outputs=model.layers[-4].output)
    # print(embedding.summary())

    anchor_in = tf.keras.Input(shape=(224,224,3))
    pos_in = tf.keras.Input(shape=(224,224,3))
    neg_in = tf.keras.Input(shape=(224,224,3))

    anchor_out = embedding(anchor_in)
    anchor_out = tf.keras.layers.BatchNormalization()(anchor_out)


    pos_out = embedding(pos_in)
    pos_out = tf.keras.layers.BatchNormalization()(pos_out)
    

    neg_out = embedding(neg_in)
    neg_out = tf.keras.layers.BatchNormalization()(neg_out)


    snnModel = tf.keras.Model(inputs=(anchor_in, pos_in, neg_in), outputs=(anchor_out,pos_out,neg_out))
    snnModel.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss=lossless_triplet_loss)
    # print(embedding.summary())
    print(snnModel.summary())
    return snnModel
