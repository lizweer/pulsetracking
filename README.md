# pulsetracking
Pulsetracking is a module for tracking and plotting EOD activity from pulsefish recorded by an electrode grid.

For tracking pulsefish EODs and location see function get_clusters() in pulsetracking.pulsetracking. If the module is called from the command line, the code under __main__ is executed. Here, get_clusters() is called on a list of input files from multiple days of recording data.

For post-processing and plotting results, see plot_traces() in pulsetracking.create_rate_traces. If the module is called from the command line, the code under __main__ is executed. Here, plot_traces() is called on a list of input files from multiple days of analysed recording data.

#### notes
- try adding a max_eps to OPTICS() in pulsetracking.analyse_window().
