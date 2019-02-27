from keras import optimizers

class iz_optimizer:

    def __init__(self,parameters):

        optimizer = None
        loss      = None
        metrics   = None

        if parameters["name"].lower() == "adam":
            self.optimizer = optimizers.Adam(**parameters["oo_params"])

        if parameters["name"].lower() == "sgd":
            self.optimizer = optimizers.SGD(**parameters["oo_params"])

        if parameters["name"].lower() == "rmsprop":
            self.optimizer = optimizers.RMSprop(**parameters["oo_params"])

        if parameters["name"].lower() == "adagrad":
            self.optimizer = optimizers.Adagrad(**parameters["oo_params"])

        if parameters["name"].lower() == "adadelta":
            self.optimizer = optimizers.Adadelta(**parameters["oo_params"])

        if parameters["name"].lower() == "adamax":
            self.optimizer = optimizers.Adamax(**parameters["oo_params"])

        if parameters["name"].lower() == "nadam":
            self.optimizer = optimizers.Nadam(**parameters["oo_params"])

        self.loss = parameters["loss"]

        self.metrics = parameters["metrics"]
