import inspect
import torchvision
print("torchvision:", torchvision.__version__)
from torchvision.io import write_video
print(inspect.signature(write_video))
print(inspect.getsource(write_video)[:1500])
