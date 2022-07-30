import pickle

import numpy as np
import torch
from tqdm.auto import tqdm

from poom import optimizers as optim
from poom.utils.visualization import Visualizer
# from problems.problem_3 import n_objectives, n_variables, objs
# from problems.problem_4 import n_objectives, n_variables, objs
from problems.problem_1 import n_objectives, n_variables, objs
# from problems.problem_2 import n_objectives, n_variables, objs
from utils.helpers import make_mesh

# from problems.dtlz import n_objectives, n_variables, objs

FIGURE_PATH = "./results"

optimizer = optim.Newton(tol=1e-7,
                         max_iteration=20,
                         objectives=objs,
                         n_objectives=n_objectives,
                         n_variables=n_variables)

init_cloud = 0.3 * torch.abs(torch.randn(size=(400, n_variables)))
# res = optimizer.find_pareto_front(init_cloud)

# with open("res_1.pl", "wb") as f:
#     pickle.dump(res, f)

with open("res_1.pl", "rb") as f:
    res = pickle.load(f)

visulizer = Visualizer(n_objectives=n_objectives)
visulizer.add_data(data=res)

mesh = make_mesh(low=-4, high=4, n_points=150, n_variables=n_variables)

im_f = []
for point in tqdm(mesh):
    im_f.append(optimizer.calc(point))
im_f = np.array(im_f).reshape(-1, n_objectives)

# with open("./res_steepest_2.pl", "rb") as f:
#     res_steepest = pickle.load(f)

# visulizer = Visualizer(n_objectives=n_objectives)
# visulizer.add_data(data=res_steepest)

fig = visulizer.plot_domain(im_f)
fig.show()
fig.write_html(f"{FIGURE_PATH}/pareto_front_2_steepest.html")

fig = visulizer.plot_moving_step(n_points=50)
fig.show()
fig.write_html(f"{FIGURE_PATH}/trace_back_2_steepest.html")

fig = visulizer.plot_value_by_step()
fig.show()
fig.write_html(f"{FIGURE_PATH}/value_trace_2_steepest.html")
