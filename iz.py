#
# IZ model
#
from images import iz_image_dataset
from optimizers import iz_optimizer
from utils import iz_load
from sklearn.model_selection import train_test_split

class izmod:

    def __init__(self):

        dataset   = None
        network     = None
        optimizer = None
        X = None
        y = None
        X_test = None
        y_test = None

    # DATASET
    def new_dataset(self,parameters):
        d = iz_image_dataset(parameters)
        self.assign_dataset(d)
        self.X = d.X
        self.y = d.y

    def infos_dataset(self):
        self.dataset.infos()

    def save_dataset(self,filename):
        self.dataset.save(filename)

    def load_dataset(self,filename):
        d = iz_load(filename)
        self.assign_dataset(d)
        self.X = d.X
        self.y = d.y

    def assign_dataset(self,dataset):
        self.dataset = dataset

    def split_dataset(self,test=0.2,random_state=42):
        X,X_test,y,y_test= train_test_split(self.X,self.y,test_size=test,random_state=random_state)
        self.X = X
        self.y = y
        self.X_test = X_test
        self.y_test = y_test
        self.dataset.X_ = X
        self.dataset.y_ = y
        self.dataset.X_test = X_test
        self.dataset.y_test = y_test


    # OPTIMIZER
    def new_optimizer(self,parameters):
        self.optimizer = iz_optimizer(parameters)

    # NETWORK
    def new_network(self,network):
        self.network = network
    def infos_network(self):
        self.network.summary()


    def compile(self):
        self.network.compile(self.optimizer.optimizer, self.optimizer.loss, self.optimizer.metrics)










    def assign_optimizer(self,optimizer):
        self.optimizer = optimizer
        self.network.compile(optimizer[0],optimizer[1],optimizer[2])

    def train(self,parameters=None):

        #Default parameters
        epochs          = 10
        batch_size      = 32
        validation_data = None
        callbacks       = None

        if "epochs" in parameters:
            epochs = parameters["epochs"]
        if "batch_size" in parameters:
            batch_size = parameters["batch_size"]
        if "validation_data" in parameters:
            validation_data = parameters["validation_data"]
        if "callbacks" in parameters:
            callbacks = parameters["callbacks"]

        history = self.network.fit(self.dataset.X_,self.dataset.y_,epochs=epochs,batch_size=batch_size,validation_data=validation_data,callbacks=callbacks)
