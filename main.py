import numpy as np
from tqdm.auto import tqdm

from poom import optimizers as optim
from problems import objs

optimizer = optim.Newton(tol=1e-5, max_iteration=100, objectives=objs)

init_cloud = np.random.uniform(low=-4, high=4, size=(50, 2))
res = []
for init in tqdm(init_cloud, leave=False):
    out = optimizer.fit(init)
    res.append(out)

res = np.array(res)
print(res)
