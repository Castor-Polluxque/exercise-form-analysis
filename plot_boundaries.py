from pathlib import Path
import csv, json
import numpy as np
import matplotlib.pyplot as plt

session = Path("/Users/nhburke/Desktop/BioSensing/drive-download-20260414T043116Z-3-001/squats/Spencer/squat_20_-2026-04-21_23-42-43")
out = session / "rep_analysis_plot_multi_only"

res = json.loads((out / "plot_multi_result.json").read_text())
bounds = list(csv.DictReader(open(out / "plot_multi_boundaries.csv")))

def read_csv(path):
    t,x,y,z=[],[],[],[]
    with open(path) as f:
        r=csv.DictReader(f)
        for row in r:
            t.append(float(row['time']))
            x.append(float(row['x']))
            y.append(float(row['y']))
            z.append(float(row['z']))
    return np.array(t)/1e9,np.array(x),np.array(y),np.array(z)

t,x,y,z = read_csv(session/"Accelerometer.csv")
t = t - t[0]
mag = np.sqrt(x*x + y*y + z*z)

fig,ax = plt.subplots(figsize=(16,6))
ax.plot(t, mag, linewidth=1)

for b in bounds:
    s = float(b['start_s'])
    e = float(b['end_s'])
    ax.axvspan(s, e, alpha=0.25)

ax.set_title("Rep Segmentation")
ax.set_xlabel("seconds")
ax.set_ylabel("accel magnitude")
ax.grid(True)

plot_path = out / "plot_multi_only.png"
fig.savefig(plot_path, dpi=180)

print("Saved:", plot_path)
