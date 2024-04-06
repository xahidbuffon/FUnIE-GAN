
import csv
import torch

all_loss = []
batch_loss = []

file_name = "v16.log"
with open(file_name) as f:
    epoch = 0
    for line in f:
        ss = line.split()
        e = int(ss[3][:-5])
        d = float(ss[5][:-1])
        g = float(ss[7][:-1])
        c = float(ss[9][:-1])
        l1 = float(ss[11][:-1])

        if(e!=epoch): # flush
            all_loss.append(((torch.Tensor(batch_loss)).mean(dim=0).tolist()))
            batch_loss = []
            epoch = e
        
        batch_loss.append([e, d, g, c, l1])

all_loss.append(((torch.Tensor(batch_loss)).mean(dim=0).tolist()))
out_file = f"{file_name[:-4]}_loss.csv"
with open(out_file, 'w') as f:
    csv.writer(f, delimiter=',').writerows(all_loss)