import streamlit as st

st.title("Customizable Neural Network App")

num_neurons = st.sidebar.slider("Number of Neurons in hidden layer:", 1, 64)
num_epochs = st.sidebar.slider("Number of Epochs for model:", 1, 10)
activation = st.sidebar.text_input("Activation type(relu, tanh etc.):")
"The number of neurons is: " + str(num_neurons)
"The number of epochs is: " + str(num_epochs)
"The activation type is: " + activation

if st.button("Train the model"):
    "We are training the model..."
    
    import tensorflow as tf
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.layers import *
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.callbacks import ModelCheckpoint
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    cols = st.columns(3)
    with cols[0]:
        st.image(x_train[0], width=100)
    with cols[1]:
        st.image(x_train[1], width=100)
    with cols[2]:
        st.image(x_train[2], width=100)


    def preprocess_images(images):
        return images/255
    x_train = preprocess_images(x_train)
    x_test = preprocess_images(x_test)

    model = Sequential()
    model.add(InputLayer((28,28)))
    model.add(Flatten())
    model.add(Dense(num_neurons, activation))  # We have to use inputs from streamlit over here
    model.add(Dense(10))
    model.add(Softmax())
    model.compile(loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
    # model.summary()

    save_cp = ModelCheckpoint('model',save_best_only = 'True')
    history_cp = tf.keras.callbacks.CSVLogger('history.csv',separator=',', append = False)
    model.fit(x_train,y_train, validation_data = (x_test, y_test), epochs = num_epochs, callbacks = [save_cp, history_cp])
    "We can now evaluate the model!"

if st.button("Evaluate the model"):
    "We are evaluating the model..."
    import matplotlib.pyplot as plt
    import pandas as pd

    history = pd.read_csv('history.csv')
    fig = plt.figure()
    plt.plot(history['epoch'], history['accuracy'])
    plt.plot(history['epoch'], history['val_accuracy'])
    plt.title("Model Accuracy across Epochs")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(['Training', 'Validation'])
    fig