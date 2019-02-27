# Create a new dataset
# Make the training
import sys, os
sys.path.append(os.path.abspath('..\\..\\izmod'))

from images import iz_image_dataset
from optimizers import iz_optimizer
from iz import izmod

from keras.models import Model
from keras.layers import Input,Conv2D,Activation,MaxPooling2D,Flatten,Dense

from keras.optimizers import Adam

# izmod model object
iz = izmod()

# IZ DATASET ------------------------------------------------------------------
# iz dataset object
parameters = {
    "path"  : "C:\\Users\\daian\\Desktop\\DATA\\bacteria\\",
    "resize": (32,32),
    "name"  : "Bacteria"
}
iz.new_dataset(parameters)
iz.infos_dataset()

iz.split_dataset(test=0.2)
iz.infos_dataset()

#iz.save_dataset("bacteria.pkl")
#iz2 = izmod()
#iz2.load_dataset("bacteria.pkl")
#iz2.infos_dataset()

# IZ OPTIMIZER ----------------------------------------------------------------
# iz optimizer object
o_params={
"name" : "Adam",
"oo_params" : {
    "lr": 1e-3,
    },
"loss" : "categorical_crossentropy",
"metrics" : ["accuracy"]
}
iz.new_optimizer(o_params)

# IZ NETWORK  ------------------------------------------------------------------
# Design the network with keras
inputs = Input(shape=iz.dataset.input_shape())
x = Conv2D(6, kernel_size = (5, 5), strides=(1,1), padding="valid",input_shape=(32, 32, 1)) (inputs)
x = Activation("relu") (x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2)) (x)
x = Conv2D(16, kernel_size = (5, 5), strides=(1,1), padding="valid") (x)
x = Activation("relu") (x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2)) (x)
x = Flatten() (x)
x = Dense(120) (x)
x = Dense(84) (x)
x = Dense(4) (x)
output = Activation("softmax") (x)
network = Model(inputs=inputs, outputs=output)
iz.new_network(network)
iz.infos_network()

# Compile network and optimizer
iz.compile()

# IZ Training  ----------------------------------------------------------------
from keras.utils import to_categorical
# from sklearn.preprocessing import MinMaxScaler
# iz.X = MinMaxScaler().fit_transform(iz.X)
iz.X = iz.X/255/255
iz.y = to_categorical(iz.y)

from sklearn.model_selection import train_test_split
X_train,X_valid,y_train,y_valid = train_test_split(iz.X,iz.y,test_size=0.2,random_state=42)

t_params={
"train_val_test" : [0.7,0.2,0.1]
}

#iz.new_training()

iz.network.fit(X_train,y_train,validation_data=(X_valid,y_valid),epochs=10)





#dataset.split(test=0.2)
#dataset.preprocess(images = "minmax", targets= "categorical")

# dataset assignment to iz object
# iz.assign_dataset(dataset)
# iz.dataset.infos()
#
# from sklearn.model_selection import train_test_split
#
# X_,X_test, y_,y_test = train_test_split(iz.dataset.X,iz.dataset.y,test_size=0.2,random_state=42)
#
# print(X_.shape)
# print(y_.shape)


# lenet5.fit(iz.X_,iz.y_,epochs=10)
