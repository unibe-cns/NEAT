

blocks_empty= dict(
parameters=
"""

parameters:
""",
state=
"""

state:
""",
equations=
"""

equations:
""",
functions=
"""
""",
inputs=
"""

inputs:
""",
output=
"""

output: spike
"""
)


blocks_default_syns = dict(
parameters=
"""

parameters:
    # synaptic parameters
    e_AMPA real = 0 mV            # Excitatory reversal Potential
    tau_r_AMPA real = 0.2 ms      # Synaptic Time Constant Excitatory Synapse
    tau_d_AMPA real = 3.0 ms      # Synaptic Time Constant Excitatory Synapse

    e_GABA real = 0 mV            # Inhibitory reversal Potential
    tau_r_GABA real = 0.2 ms      # Synaptic Time Constant Inhibitory Synapse
    tau_d_GABA real = 3.0 ms      # Synaptic Time Constant Inhibitory Synapse

    e_NMDA real = 0 mV            # NMDA reversal Potential
    tau_r_NMDA real = 0.2 ms      # Synaptic Time Constant NMDA Synapse
    tau_d_NMDA real = 3.0 ms      # Synaptic Time Constant NMDA Synapse

    e_AN_AMPA real = 0 mV            # Excitatory reversal Potential
    tau_r_AN_AMPA real = 0.2 ms      # Synaptic Time Constant Excitatory Synapse
    tau_d_AN_AMPA real = 3.0 ms      # Synaptic Time Constant Excitatory Synapse
    e_AN_NMDA real = 0 mV            # NMDA reversal Potential
    tau_r_AN_NMDA real = 0.2 ms      # Synaptic Time Constant NMDA Synapse
    tau_d_AN_NMDA real = 3.0 ms      # Synaptic Time Constant NMDA Synapse
    NMDA_ratio real = 2.0      # NMDA_ratio
""",
state=
"""

state:
    v_comp real
""",
equations=
"""

equations:
    kernel g_AMPA = g_norm_AMPA * ( - exp(-t / tau_r_AMPA) + exp(-t / tau_d_AMPA) )
    inline AMPA real = convolve(g_AMPA, spikes_AMPA) * (e_AMPA - v_comp)

    kernel g_GABA = g_norm_GABA * ( - exp(-t / tau_r_GABA) + exp(-t / tau_d_GABA) )
    inline GABA real = convolve(g_GABA, spikes_GABA) * (e_GABA - v_comp )

    kernel g_NMDA = g_norm_NMDA * ( - exp(-t / tau_r_NMDA) + exp(-t / tau_d_NMDA) )
    inline NMDA real = convolve(g_NMDA, spikes_NMDA) * (e_NMDA - v_comp ) / (1. + 0.3 * exp( -.1 * v_comp ))

    kernel g_AN_AMPA = g_norm_AN_AMPA * ( - exp(-t / tau_r_AN_AMPA) + exp(-t / tau_d_AN_AMPA) )
    kernel g_AN_NMDA = g_norm_AN_NMDA * ( - exp(-t / tau_r_AN_NMDA) + exp(-t / tau_d_AN_NMDA) )
    inline AMPA_NMDA real = convolve(g_AN_AMPA, spikes_AN) * (e_AN_AMPA - v_comp) + NMDA_ratio * \
                            convolve(g_AN_NMDA, spikes_AN) * (e_AN_NMDA - v_comp) / (1. + 0.3 * exp( -.1 * v_comp ))
""",
functions=
"""
""",
inputs=
"""

inputs:
    spikes_AMPA uS <- spike
    spikes_GABA uS <- spike
    spikes_NMDA uS <- spike
    spikes_AN uS <- spike
""",
output=
"""

output: spike
"""
)


def writeNestmlBlocks(blocks, path_name, neuron_name,
                      write_blocks=['parameters', 'state', 'equations',
                                    'inputs', 'output', 'functions']):
    for block, blockstr in blocks.items():
        if block != 'functions':
            blocks[block] = blockstr + 'end\n\n'


    fname = os.path.join(path_name, neuron_name + ".nestml")

    file = open(fname, 'w')
    file.write('\nneuron %s:\n'%neuron_name)
    for blockstr in blocks.values():
        file.write(blockstr)
    file.write('\nend')
    file.close()

    return fname
