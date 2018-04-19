
# This is a hack to allow running headless e.g. Jenkins
import os
if not os.environ.get('DISPLAY'):
    import matplotlib
    matplotlib.use('Agg')