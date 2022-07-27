import numpy as np
import torch
from tqdm.auto import tqdm

from poom import optimizers as optim
from poom.optimizers.newton.utils import Visualizer
# from problems.problem_2 import n_objectives, n_variables, objs

# from problems.dtlz import n_objectives, n_variables, objs

# from problems.problem_3 import n_objectives, n_variables, objs
from problems.problem_1 import n_objectives, n_variables, objs

FIGURE_PATH = "./results"

optimizer = optim.Newton(tol=1e-12,
                         max_iteration=50,
                         objectives=objs,
                         n_objectives=n_objectives,
                         n_variables=n_variables)

init_cloud = torch.randn(size=(400, n_variables))
res = optimizer.find_pareto_front(init_cloud)
visulizer = Visualizer(results=res, n_objectives=n_objectives)

sample = np.linspace(-3, 3, 51).reshape(-1, 1)
xv, yv, zv = np.meshgrid(sample, sample, sample)
im_f = []
for _x, _y, _z in tqdm(
        zip(xv.reshape(-1, 1), yv.reshape(-1, 1), zv.reshape(-1, 1))):
    im_f.append(optimizer.calc([_x, _y, _z]))
im_f = np.array(im_f).reshape(-1, n_objectives)

fig = visulizer.plot_domain(im_f)
fig.show()
fig.write_html(f"{FIGURE_PATH}/pareto_front_3_newton.html")

fig = visulizer.plot_moving_step(n_points=30)
fig.show()
fig.write_html(f"{FIGURE_PATH}/trace_back_3_newton.html")

fig = visulizer.plot_value_by_step()
fig.show()
fig.write_html(f"{FIGURE_PATH}/value_trace_3_newton.html")
