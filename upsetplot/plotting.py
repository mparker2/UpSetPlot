from __future__ import print_function, division, absolute_import

import itertools
from functools import reduce
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.tight_layout import get_renderer

from .data import from_sets


def _process_data(data, sort_by, sort_sets_by,
                  bootstrap_exp, global_set_size,
                  ci, nboot, discard_empty):
    # check all indices are vertical
    assert all(set([True, False]) >= set(level) for level in data.index.levels)
    if not data.index.is_unique:
        data = (data
                .groupby(level=list(range(data.index.nlevels)))
                .sum())

    totals = []
    for i in range(data.index.nlevels):
        idxslice = pd.IndexSlice[(slice(None),) * i + (True,)]
        # FIXME: can get IndexingError if level only contains False
        totals.append(data.loc[idxslice].sum())
    totals = pd.Series(totals, index=data.index.names)
    data, totals = _order_data(data, totals, sort_by, sort_sets_by)
    if discard_empty:
        data = data[data.astype(bool)]

    if bootstrap_exp:
        if global_set_size is None:
            global_set_size = np.nansum(data.values)
        exp_med, exp_err = _bootstrap_expected(
            totals, global_set_size, ci, nboot)
        exp_med = exp_med[data.index]
        exp_err = exp_err[data.index]
    else:
        exp_med, exp_err = None, None

    return data, totals, exp_med, exp_err


def _order_data(data, totals, sort_by, sort_sets_by):
    
    if sort_sets_by == 'cardinality':
        totals.sort_values(ascending=False, inplace=True)

    elif sort_sets_by is not None:
        raise ValueError('Unknown sort_sets_by: %r' % sort_sets_by)

    data = data.reorder_levels(totals.index.values)

    if sort_by == 'cardinality':
        data = data.sort_values(ascending=False)
    elif sort_by == 'degree':
        comb = itertools.combinations
        o = pd.DataFrame([{name: True for name in names}
                          for i in range(data.index.nlevels + 1)
                          for names in comb(data.index.names, i)],
                         columns=data.index.names)
        o.fillna(False, inplace=True)
        o = o.astype(bool)
        o.set_index(data.index.names, inplace=True)
        # FIXME: should use reindex(index=...) ??
        data = data.loc[o.index]
    else:
        raise ValueError('Unknown sort_by: %r' % sort_by)
    return data, totals


def _set_intersections(sets):
    for i in range(1, len(sets) + 1):
        yield from itertools.combinations(sets, i)


def _confidence_interval(a, which=95):
    p = 50 - which / 2, 50 + which / 2
    l, u = np.percentile(a, p)
    return np.median(a) - l, u - np.median(a)


def _bootstrap_expected(totals, global_set_size, ci, nboot):
    global_set = np.arange(global_set_size)
    intersections = list(_set_intersections(totals.index.values))
    exp_overlaps = []
    for _ in range(nboot):
        sampled_sets = OrderedDict()
        for set_name, set_size in totals.iteritems():
            sampled_sets[set_name] = set(np.random.choice(
                global_set, size=set_size, replace=False))
        sampled_sets['global'] = set(global_set)
        agg = from_sets(sampled_sets)
        agg = agg.groupby(agg.index.names[:-1]).sum()
        exp_overlaps.append(agg)

    ci_func = np.std if ci == 'std' else lambda x: _confidence_interval(x, ci)
    exp_overlaps = pd.concat(exp_overlaps, axis=1)
    exp_med = exp_overlaps.median(axis=1)
    exp_err = exp_overlaps.apply(ci_func, axis=1, raw=True)
    return exp_med, exp_err      


class _Transposed:
    """Wrap an object in order to transpose some plotting operations

    Attributes of obj will be mapped.
    Keyword arguments when calling obj will be mapped.

    The mapping is not recursive: callable attributes need to be _Transposed
    again.
    """

    def __init__(self, obj):
        self.__obj = obj

    def __getattr__(self, key):
        return getattr(self.__obj, self._NAME_TRANSPOSE.get(key, key))

    def __call__(self, *args, **kwargs):
        return self.__obj(*args, **{self._NAME_TRANSPOSE.get(k, k): v
                                    for k, v in kwargs.items()})

    _NAME_TRANSPOSE = {
        'width': 'height',
        'height': 'width',
        'hspace': 'wspace',
        'wspace': 'hspace',
        'hlines': 'vlines',
        'vlines': 'hlines',
        'bar': 'barh',
        'barh': 'bar',
        'yerr': 'xerr',
        'xerr': 'yerr',
        'xaxis': 'yaxis',
        'yaxis': 'xaxis',
        'left': 'bottom',
        'right': 'top',
        'top': 'right',
        'bottom': 'left',
        'sharex': 'sharey',
        'sharey': 'sharex',
        'get_figwidth': 'get_figheight',
        'get_figheight': 'get_figwidth',
        'set_figwidth': 'set_figheight',
        'set_figheight': 'set_figwidth',
    }


def _transpose(obj):
    if isinstance(obj, str):
        return _Transposed._NAME_TRANSPOSE.get(obj, obj)
    return _Transposed(obj)


def _identity(obj):
    return obj


class UpSet:
    """Manage the data and drawing for a basic UpSet plot

    Primary public method is :meth:`plot`.

    Parameters
    ----------
    data : pandas.Series
        Values for each set to plot.
        Should have multi-index where each level is binary,
        corresponding to set membership.
    bootstrap_expected: bool
        Whether to plot expected intersection size bars using global
        set size parameter (Default False).
    global_set_size : int or None
        Size of whole population sets are drawn from. If None,
        all examples are assumed to be present in union of sets.
        Only required for computing expected bootstraps.
    orientation : {'horizontal' (default), 'vertical'}
        If horizontal, intersections are listed from left to right.
    sort_by : {'cardinality', 'degree'}
        If 'cardinality', set intersections are listed from largest to
        smallest value.
        If 'degree', they are listed in order of the number of sets
        intersected.
    sort_sets_by : {'cardinality', None}
        Whether to sort the overall sets by total cardinality, or leave them
        in the provided order.
    facecolor : str
        Color for bar charts and dots.
    with_lines : bool
        Whether to show lines joining dots in the matrix, to mark multiple sets
        being intersected.
    element_size : float or None
        Side length in pt. If None, size is estimated to fit figure
    intersections_plot_elements : int
        The intersections plot should be large enough to fit this many matrix
        elements.
    totals_plot_elements : int
        The totals plot should be large enough to fit this many matrix
        elements.
    """

    def __init__(self, data, bootstrap_expected=False,
                 global_set_size=None, ci=95, n_boot=100,
                 orientation='horizontal', sort_by='degree',
                 sort_sets_by='cardinality', discard_empty=False,
                 facecolor='black', with_lines=True, element_size=32,
                 intersection_plot_elements=6, totals_plot_elements=2):

        self._bootstrap_expected = bootstrap_expected
        self._err_type = 'std' if ci == 'std' else 'ci'
        self._horizontal = orientation == 'horizontal'
        self._reorient = _identity if self._horizontal else _transpose
        self._facecolor = facecolor
        self._with_lines = with_lines
        self._element_size = element_size
        self._totals_plot_elements = totals_plot_elements
        self._intersection_plot_elements = intersection_plot_elements

        (self.intersections, self.totals, self.exp_med,
         self.exp_err) = _process_data(data,
                                       sort_by=sort_by,
                                       sort_sets_by=sort_sets_by,
                                       bootstrap_exp=bootstrap_expected,
                                       global_set_size=global_set_size,
                                       ci=ci, nboot=n_boot,
                                       discard_empty=discard_empty)
        if not self._horizontal:
            self.intersections = self.intersections[::-1]
            if bootstrap_expected:
                self.exp_med = self.exp_med[::-1]
                self.exp_err = self.exp_err[::-1]

    def _swapaxes(self, x, y):
        if self._horizontal:
            return x, y
        return y, x

    def make_grid(self, fig=None):
        """Get a SubplotSpec for each Axes, accounting for label text width
        """
        n_cats = len(self.totals)
        n_inters = len(self.intersections)

        if fig is None:
            fig = plt.gcf()

        # Determine text size to determine figure size / spacing
        r = get_renderer(fig)
        t = fig.text(0, 0, '\n'.join(self.totals.index.values))
        textw = t.get_window_extent(renderer=r).width
        t.remove()

        MAGIC_MARGIN = 10  # FIXME
        figw = self._reorient(fig.get_window_extent(renderer=r)).width
        if self._element_size is None:
            colw = (figw - textw - MAGIC_MARGIN) / (len(self.intersections) +
                                                    self._totals_plot_elements)
        else:
            fig = self._reorient(fig)
            render_ratio = figw / fig.get_figwidth()
            colw = self._element_size / 72 * render_ratio
            figw = (colw * (len(self.intersections) +
                            self._totals_plot_elements) +
                    MAGIC_MARGIN + textw)
            fig.set_figwidth(figw / render_ratio)
            fig.set_figheight((colw * (n_cats +
                                       self._intersection_plot_elements)) /
                              render_ratio)

        text_nelems = int(np.ceil(figw / colw - (len(self.intersections) +
                                                 self._totals_plot_elements)))

        GS = self._reorient(matplotlib.gridspec.GridSpec)
        gridspec = GS(*self._swapaxes(n_cats +
                                      self._intersection_plot_elements,
                                      n_inters + text_nelems +
                                      self._totals_plot_elements),
                      hspace=1)
        if self._horizontal:
            return {'intersections': gridspec[:-n_cats, -n_inters:],
                    'matrix': gridspec[-n_cats:, -n_inters:],
                    'shading': gridspec[-n_cats:, :],
                    'totals': gridspec[-n_cats:, :self._totals_plot_elements],
                    'gs': gridspec}
        else:
            return {'intersections': gridspec[-n_inters:, n_cats:],
                    'matrix': gridspec[-n_inters:, :n_cats],
                    'shading': gridspec[:, :n_cats],
                    'totals': gridspec[:self._totals_plot_elements, :n_cats],
                    'gs': gridspec}

    def plot_matrix(self, ax):
        """Plot the matrix of intersection indicators onto ax
        """
        ax = self._reorient(ax)
        data = self.intersections
        n_sets = data.index.nlevels

        idx = np.flatnonzero(data.index.to_frame()[data.index.names].values)
        c = np.array(['lightgrey'] * len(data) * n_sets, dtype='O')
        c[idx] = self._facecolor
        x = np.repeat(np.arange(len(data)), n_sets)
        y = np.tile(np.arange(n_sets), len(data))
        if self._element_size is not None:
            s = (self._element_size * .35) ** 2
        else:
            # TODO: make s relative to colw
            s = 200
        ax.scatter(*self._swapaxes(x, y), c=c.tolist(), linewidth=0, s=s)

        if self._with_lines:
            line_data = (pd.Series(y[idx], index=x[idx])
                         .groupby(level=0)
                         .aggregate(['min', 'max']))
            ax.vlines(line_data.index.values,
                      line_data['min'], line_data['max'],
                      lw=2, colors=self._facecolor)

        tick_axis = ax.yaxis
        tick_axis.set_ticks(np.arange(n_sets))
        tick_axis.set_ticklabels(data.index.names,
                                 rotation=0 if self._horizontal else -90)
        ax.xaxis.set_visible(False)
        ax.tick_params(axis='both', which='both', length=0)
        if not self._horizontal:
            ax.yaxis.set_ticks_position('top')
        ax.set_frame_on(False)

    def plot_intersections(self, ax):
        """Plot bars indicating intersection size
        """
        
        ax = self._reorient(ax)
        width = 0.25 if self._bootstrap_expected else 0.5
        x = np.arange(len(self.intersections))
        obs_x = x - 0.125 if self._bootstrap_expected else x
        exp_x = x + 0.125
        if self.exp_err is not None:
            if self._err_type == 'std':
                exp_err = self.exp_err.values
            else:
                exp_err = np.array(self.exp_err.values.tolist()).T
        ax.bar(obs_x, self.intersections,
               width, color=self._facecolor, zorder=10, align='center')
        if self._bootstrap_expected:
            ax.bar(exp_x, self.exp_med, width,
                   color='lightgrey',
                   align='center',
                   **{self._reorient('yerr'): exp_err})
        ax.xaxis.set_visible(False)
        for x in ['top', 'bottom', 'right']:
            ax.spines[self._reorient(x)].set_visible(False)

        tick_axis = ax.yaxis
        tick_axis.grid(True)
        tick_axis.set_label('Intersection size')
        # tick_axis.set_tick_params(direction='in')

    def plot_totals(self, ax):
        """Plot bars indicating total set size
        """
        orig_ax = ax
        ax = self._reorient(ax)
        ax.barh(np.arange(len(self.totals.index.values)), self.totals,
                .5, color=self._facecolor, align='center')
        max_total = self.totals.max()
        if self._horizontal:
            orig_ax.set_xlim(max_total, 0)
        for x in ['top', 'left', 'right']:
            ax.spines[self._reorient(x)].set_visible(False)
        ax.yaxis.set_visible(False)
        ax.xaxis.grid(True)
        ax.patch.set_visible(False)

    def plot_shading(self, ax):
        # alternating row shading (XXX: use add_patch(Rectangle)?)
        for i in range(0, len(self.totals), 2):
            rect = plt.Rectangle(self._swapaxes(0, i - .4),
                                 *self._swapaxes(*(1, .8)),
                                 facecolor='#f5f5f5', lw=0, zorder=0)
            ax.add_patch(rect)
        ax.set_frame_on(False)
        ax.tick_params(
            axis='both',
            which='both',
            left='off',
            right='off',
            bottom='off',
            top='off',
            labelbottom='off',
            labelleft='off')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    def plot(self, fig=None):
        """Draw all parts of the plot onto fig or a new figure

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            Defaults to a new figure.

        Returns
        -------
        subplots : dict of matplotlib.axes.Axes
            Keys are 'matrix', 'intersections', 'totals', 'shading'
        """
        if fig is None:
            fig = plt.figure(figsize=(10, 6))
        specs = self.make_grid(fig)
        shading_ax = fig.add_subplot(specs['shading'])
        self.plot_shading(shading_ax)
        matrix_ax = self._reorient(fig.add_subplot)(specs['matrix'],
                                                    sharey=shading_ax)
        self.plot_matrix(matrix_ax)
        inters_ax = self._reorient(fig.add_subplot)(specs['intersections'],
                                                    sharex=matrix_ax)
        self.plot_intersections(inters_ax)
        totals_ax = self._reorient(fig.add_subplot)(specs['totals'],
                                                    sharey=matrix_ax)
        self.plot_totals(totals_ax)
        return {'matrix': matrix_ax,
                'intersections': inters_ax,
                'shading': shading_ax,
                'totals': totals_ax}


def plot(data, fig=None, **kwargs):
    """Make an UpSet plot of data on fig

    Parameters
    ----------
    data : pandas.Series
        Values for each set to plot.
        Should have multi-index where each level is binary,
        corresponding to set membership.
    fig : matplotlib.figure.Figure, optional
        Defaults to a new figure.
    kwargs
        Other arguments for :class:`UpSet`

    Returns
    -------
    subplots : dict of matplotlib.axes.Axes
        Keys are 'matrix', 'intersections', 'totals', 'shading'
    """
    return UpSet(data, **kwargs).plot(fig)
