import sys, os
sys.path.append(os.path.abspath('..\\..\\izmod'))

from images import iz_image_dataset
from utils import iz_load

parameters = {
    "path"  : "C:\\Users\\daian\\Desktop\\DATA\\bacteria\\",
    "resize": (32,32),
    "name"  : "Bacteria"
}

iz = iz_image_dataset(parameters)

iz.split(test=0.2)

iz.preprocess(images = "minmax", targets= "categorical")

iz.infos()

print(iz.output_neurons())
