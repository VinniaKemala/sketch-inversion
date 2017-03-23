## based on https://github.com/dmlc/mxnet/issues/1302
## Parses the model fit log file and generates a train/val vs epoch plot
import matplotlib.pyplot as plt
import numpy as np
import re
import argparse

parser = argparse.ArgumentParser(description='Parses log file and generates train/val curves')
parser.add_argument('--log-file', type=str,default="log/log_tr_sketch",
                    help='the path of log file')
args = parser.parse_args()

print('Parsing...')
TR_RE = re.compile('\sEpoch Data Grad:\s([\d\.]+)')
log = open(args.log_file).read()
log_tr = np.array([float(x) for x in TR_RE.findall(log)])
idx = np.arange(len(log_tr))

# Plotting
plt.figure(figsize=(8, 6))
plt.suptitle('Gradient error', fontsize=14, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Data gradient")

plt.plot(idx, log_tr, 'o', linestyle='-', color="r",
         label="Gradient err")

plt.legend(loc="best")
plt.xticks(np.arange(min(idx), max(idx)+1, 20))

# Change the min and max values of Y if you can't see the curve
plt.yticks(np.arange(50, 80, 2)) # min 50, max 80
plt.ylim([50, 80])
plt.show()
