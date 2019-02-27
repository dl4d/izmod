import sys, os
sys.path.append(os.path.abspath('..\\..\\izmod'))

from images import iz_image_dataset
from iz import iz

from keras.models import Model
from keras.layers import Input,Conv2D,Activation,MaxPooling2D,Flatten,Dense

from keras.optimizers import Adam

# iz model object
iz = iz()

# iz dataset object
parameters = {
    "path"       : "C:\\Users\\daian\\Desktop\\DATA\\bacteria\\",
    "resize"     : (32,32),
    "name"       : "Bacteria",
    "test_split" : 0.2
}
dataset = iz_image_dataset(parameters)
dataset.split(test=0.2)
dataset.preprocess(images = "minmax", targets= "categorical")

# dataset assignment to iz object
iz.assign_dataset(dataset)
iz.dataset.infos()


# Design the network with keras
inputs = Input(shape=dataset.input_shape())
x = Conv2D(6, kernel_size = (5, 5), strides=(1,1), padding="valid",input_shape=(32, 32, 1)) (inputs)
x = Activation("relu") (x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2)) (x)
x = Conv2D(16, kernel_size = (5, 5), strides=(1,1), padding="valid") (x)
x = Activation("relu") (x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2)) (x)
x = Flatten() (x)
x = Dense(120) (x)
x = Dense(84) (x)
x = Dense(dataset.output_neurons()) (x)
output = Activation("softmax") (x)
network = Model(inputs=inputs, outputs=output)

# Assign network
iz.assign_network(network)
iz.network.summary()

# Design optimizer with Keras
optimizer = Adam(lr=1e-5)
loss      = "categorical_crossentropy"
metrics   = ["accuracy"]

iz.assign_optimizer((optimizer,loss,metrics))

from keras.utils import to_categorical

print(iz.dataset.X_scalers_type)


parameters={
"epochs" : 100,
"batch_size" : 32,
"validation_data": (iz.dataset.X_scalers[0].fit_transform(iz.dataset.X_test),to_categorical(iz.dataset.y_test)),
"callbacks" : None,
}

iz.train(parameters)




# lenet5.fit(iz.X_,iz.y_,epochs=10)
