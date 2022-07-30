import os
import csv

def log_bin(bins, partition, bin_DIR):
    os.makedirs(bin_DIR, exist_ok=True)
    bin_PATH = os.path.join(bin_DIR, 'bin.csv')

    bin_file = open(bin_PATH, "w")
    bin_writer = csv.writer(bin_file)

    for idx, bin in enumerate(bins):
        sz = len(partition[idx])
        row = [idx, bin, sz]
        bin_writer.writerow(row)
    bin_file.close()


def save_model(model, model_DIR):
    save_PATH = os.path.join(model_DIR, 'save.pt')
    torch.save(model.state_dict(), save_PATH)