import holoviews as hv
import pandas as pd
import panel as pn
from holoviews import opts

#hv.renderer('bokeh').theme = 'light_minimal' # Graph style
hv.extension('bokeh')
pn.extension()

plot_opts = {'height': 400, 'width': 600, 'line_width':2.5, 'shared_axes':False, 'fontscale':1.2}
opts.defaults(opts.Curve(**plot_opts), opts.Scatter(**plot_opts))
