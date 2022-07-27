from typing import List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .report import ResultReporter


class Visualizer:

    def __init__(self,
                 results: List[ResultReporter],
                 n_objectives: int = 2) -> None:
        self.results = results
        self.n_objectives = n_objectives
        y_min = [r[-1] for r in self.results]
        self.pareto_front = np.array(y_min)
        self.trace = np.array(results)

    def plot_pareto_front(self):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=self.pareto_front[:, 0],
                       y=self.pareto_front[:, 1],
                       mode="markers"))
        return fig

    def __plot_domain_2d(self, domain):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=domain[:, 0],
                       y=domain[:, 1],
                       mode="markers",
                       marker=dict(color='red')))
        fig.add_trace(
            go.Scatter(x=self.pareto_front[:, 0],
                       y=self.pareto_front[:, 1],
                       mode="markers",
                       marker=dict(color='teal')))
        return fig

    def __plot_domain_3d(self, domain):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(x=domain[:, 0],
                         y=domain[:, 1],
                         z=domain[:, 2],
                         mode="markers",
                         marker=dict(color='red', size=1),
                         name="Im F"))
        fig.add_trace(
            go.Scatter3d(x=self.pareto_front[:, 0],
                         y=self.pareto_front[:, 1],
                         z=self.pareto_front[:, 2],
                         mode="markers",
                         marker=dict(color='blue', size=4),
                         name="Pareto Points"))
        return fig

    def plot_domain(self, domain):
        if self.n_objectives == 2:
            fig = self.__plot_domain_2d(domain)
        elif self.n_objectives == 3:
            fig = self.__plot_domain_3d(domain)
        return fig

    def __plot_moving_step_2d(self, n_points: int = None):
        if n_points is None:
            n_points = len(self.trace)
        fig = go.Figure()
        for i in range(n_points):
            r = np.array(self.trace[i])
            fig.add_trace(
                go.Scatter(x=r[:, 0], y=r[:, 1], mode="lines+markers"))
        return fig

    def __plot_moving_step_3d(self, n_points: int = None):
        if n_points is None:
            n_points = len(self.trace)
        fig = go.Figure()
        for i in range(n_points):
            r = np.array(self.trace[i])
            fig.add_trace(
                go.Scatter3d(x=r[:, 0],
                             y=r[:, 1],
                             z=r[:, 2],
                             mode="lines+markers"))
        return fig

    def plot_moving_step(self, n_points: int = None):
        if n_points is None or n_points > len(self.trace):
            n_points = len(self.trace)
        if self.n_objectives == 2:
            fig = self.__plot_moving_step_2d(n_points=n_points)
        elif self.n_objectives == 3:
            fig = self.__plot_moving_step_3d(n_points=n_points)
        return fig

    def plot_value_by_step(self, index: int = 0):
        r = np.array(self.trace[index])
        n_points = len(r)
        fig = make_subplots(rows=1, cols=self.n_objectives)
        x_axis = list(range(n_points))
        for obj_idx in range(self.n_objectives):
            fig.add_trace(go.Scatter(x=x_axis,
                                     y=r[:, obj_idx],
                                     mode="lines+markers",
                                     name=f"Objective {obj_idx + 1}",
                                     marker=dict(size=1)),
                          row=1,
                          col=obj_idx + 1)
        return fig
