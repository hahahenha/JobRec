# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from . import Filter  # prevent circular import in Python < 3.5


class Self_define(Filter):
    def __init__(self, G, tau=10, normalize=False, **kwargs):

        try:
            iter(tau)
        except TypeError:
            tau = [tau]

        def kernel(x, t):
            # if t < 0:
            #     return (1 - t * x) / (G.lmax+1) * np.exp((1 - t * x) / (G.lmax+1))
            # else:
            #     return (G.lmax+1) / (1 + t * x) * np.exp((-1 - t * x) / (G.lmax+1))
            if t < 0:
                return (x + 1) / (G.lmax + 1) * np.exp(- t *(x+1) / (G.lmax + 1))
            else:
                return (G.lmax + 1) / (x + 1) * np.exp(- t * (x + 1) / (G.lmax + 1))

        g = []
        for t in tau:
            norm = np.linalg.norm(kernel(G.e, t)) if normalize else 1
            g.append(lambda x, t=t, norm=norm: kernel(x, t) / norm)

        super(Self_define, self).__init__(G, g, **kwargs)
