import tensorflow as tf
tf.random.set_seed(999)
from tensorflow.keras.regularizers import L2

def modelling(lamda):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer = L2(lamda), kernel_initializer = 'he_normal'), 
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1, activation='linear', kernel_regularizer = L2(lamda), kernel_initializer = 'he_normal')
    ])
    return model 

a = modelling(0.01)

# def nn2(lamda):
#     model = tf.keras.Sequential([
#         tf.keras.layers.Dense(32, activation='relu', kernel_regularizer = L2(lamda), kernel_initializer = 'he_normal'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dense(16, activation='relu', kernel_regularizer = L2(lamda), kernel_initializer = 'he_normal'), 
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dense(1, activation='linear', kernel_regularizer = L2(lamda), kernel_initializer = 'he_normal')
#     ])
#     return model 


# def nn3(lamda):
#     model = tf.keras.Sequential([
#         tf.keras.layers.Dense(32, activation='relu', kernel_regularizer = L2(lamda), kernel_initializer = 'he_normal'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dense(16, activation='relu', kernel_regularizer = L2(lamda), kernel_initializer = 'he_normal'), 
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dense(8, activation='relu', kernel_regularizer = L2(lamda), kernel_initializer = 'he_normal'), 
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dense(1, activation='linear', kernel_regularizer = L2(lamda), kernel_initializer = 'he_normal')
#     ])
#     return model 

# def nn4(lamda):
#     model = tf.keras.Sequential([
#         tf.keras.layers.Dense(32, activation='relu', kernel_regularizer = L2(lamda), kernel_initializer = 'he_normal'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dense(16, activation='relu', kernel_regularizer = L2(lamda), kernel_initializer = 'he_normal'), 
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dense(8, activation='relu', kernel_regularizer = L2(lamda), kernel_initializer = 'he_normal'), 
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dense(4, activation='relu', kernel_regularizer = L2(lamda), kernel_initializer = 'he_normal'), 
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dense(1, activation='linear', kernel_regularizer = L2(lamda), kernel_initializer = 'he_normal')
#     ])
#     return model 

# def nn5(lamda):
#     model = tf.keras.Sequential([
#         tf.keras.layers.Dense(32, activation='relu', kernel_regularizer = L2(lamda), kernel_initializer = 'he_normal'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dense(16, activation='relu', kernel_regularizer = L2(lamda), kernel_initializer = 'he_normal'), 
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dense(8, activation='relu', kernel_regularizer = L2(lamda), kernel_initializer = 'he_normal'), 
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dense(4, activation='relu', kernel_regularizer = L2(lamda), kernel_initializer = 'he_normal'), 
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dense(2, activation='relu', kernel_regularizer = L2(lamda), kernel_initializer = 'he_normal'), 
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dense(1, activation='linear', kernel_regularizer = L2(lamda), kernel_initializer = 'he_normal')
#     ])
#     return model 




