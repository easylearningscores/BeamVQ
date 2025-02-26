from GNOT import GNOT
import torch

torch.cuda.set_device(7)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

a = torch.zeros([16,10,64,64]).to(device)

model = GNOT(branch_sizes = [2]).to(device)
pred = model(a.permute([0,2,3,1]))
print("GNOT(2023)", pred.shape)
