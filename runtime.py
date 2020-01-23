#The runtime script measure the GPU/CPU runtime of models. 
import time
from torchprofile import profile_macs
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
inputs = torch.randn(1, 3, 224, 224, requires_grad=True).to(device)
def run_model(name, model):
    model.to(device)
    model.eval()
    print("---------------------------", name)
    macs = profile_macs(model, inputs)
    print("MACs=", macs)
    print("Prameters=", sum([param.nelement() for param in model.parameters()]))
    #torch.cuda.synchronize()
    s_time = time.time()
    output = model(inputs)
    #torch.cuda.synchronize()
    e_time = time.time()
    print("exec time=", (e_time - s_time))
    print("output=", output.type(), output.shape)

#e.g., 
import torchvision.models as models
model = models.resnet101(pretrained = False)
run_model("resnet101:", model)