import os
import copy
import warnings

BLOCKS_EMPTY = dict(
parameters=
"""
""",
state=
"""
""",
equations=
"""
""",
functions=
"""
""",
internals=
"""
""",
input=
"""
""",
output=
"""
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

    e_GABA real = -80 mV            # Inhibitory reversal Potential
    tau_r_GABA real = 0.2 ms      # Synaptic Time Constant Inhibitory Synapse
    tau_d_GABA real = 10.0 ms      # Synaptic Time Constant Inhibitory Synapse

    e_NMDA real = 0 mV            # NMDA reversal Potential
    tau_r_NMDA real = 0.2 ms      # Synaptic Time Constant NMDA Synapse
    tau_d_NMDA real = 3.0 ms      # Synaptic Time Constant NMDA Synapse

    e_AN_AMPA real = 0 mV            # Excitatory reversal Potential
    tau_r_AN_AMPA real = 0.2 ms      # Synaptic Time Constant Excitatory Synapse
    tau_d_AN_AMPA real = 3.0 ms      # Synaptic Time Constant Excitatory Synapse
    e_AN_NMDA real = 0 mV            # NMDA reversal Potential
    tau_r_AN_NMDA real = 0.2 ms      # Synaptic Time Constant NMDA Synapse
    tau_d_AN_NMDA real = 43.0 ms     # Synaptic Time Constant NMDA Synapse
    NMDA_ratio real = 2.0      # NMDA_ratio
""",
state=
"""

state:
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
    inline AMPA_NMDA real = convolve(g_AN_AMPA, spikes_AN) * (e_AN_AMPA - v_comp) + NMDA_ratio * \\
                            convolve(g_AN_NMDA, spikes_AN) * (e_AN_NMDA - v_comp) / (1. + 0.3 * exp( -.1 * v_comp ))
""",
functions=
"""
""",
internals=
"""

internals:
    tp_AMPA real = (tau_r_AMPA * tau_d_AMPA) / (tau_d_AMPA - tau_r_AMPA) * ln( tau_d_AMPA / tau_r_AMPA )
    g_norm_AMPA real =  1. / ( -exp( -tp_AMPA / tau_r_AMPA ) + exp( -tp_AMPA / tau_d_AMPA ) )

    tp_GABA real = (tau_r_GABA * tau_d_GABA) / (tau_d_GABA - tau_r_GABA) * ln( tau_d_GABA / tau_r_GABA )
    g_norm_GABA real =  1. / ( -exp( -tp_GABA / tau_r_GABA ) + exp( -tp_GABA / tau_d_GABA ) )

    tp_NMDA real = (tau_r_NMDA * tau_d_NMDA) / (tau_d_NMDA - tau_r_NMDA) * ln( tau_d_NMDA / tau_r_NMDA )
    g_norm_NMDA real =  1. / ( -exp( -tp_NMDA / tau_r_NMDA ) + exp( -tp_NMDA / tau_d_NMDA ) )

    tp_AN_AMPA real = (tau_r_AN_AMPA * tau_d_AN_AMPA) / (tau_d_AN_AMPA - tau_r_AN_AMPA) * ln( tau_d_AN_AMPA / tau_r_AN_AMPA )
    g_norm_AN_AMPA real =  1. / ( -exp( -tp_AN_AMPA / tau_r_AN_AMPA ) + exp( -tp_AN_AMPA / tau_d_AN_AMPA ) )

    tp_AN_NMDA real = (tau_r_AN_NMDA * tau_d_AN_NMDA) / (tau_d_AN_NMDA - tau_r_AN_NMDA) * ln( tau_d_AN_NMDA / tau_r_AN_NMDA )
    g_norm_AN_NMDA real =  1. / ( -exp( -tp_AN_NMDA / tau_r_AN_NMDA ) + exp( -tp_AN_NMDA / tau_d_AN_NMDA ) )
""",
input=
"""

input:
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


def stripVComp(contents):
    ss = "v_comp real"
    idx = None
    for ll, line in enumerate(contents):
        if ss in line:
            idx = ll

    if idx is not None:
        del contents[idx]


def stripComments(contents):
    for ll, line in enumerate(contents):
        try:
            s0 = line.index('#')
            contents[ll] = line[:s0] + '\n'
        except ValueError as e:
            # do nothing if there is no comment
            pass


def _getIndexOfBlock(contents, block_name):
    found, kk = False, 0
    while not found and kk < len(contents):
        if block_name in contents[kk]:
            found = True
        else:
            kk += 1


    return kk


def getBlockString(contents, block_name):
    try:
        c0 = _getIndexOfBlock(contents, block_name)
        s0 = contents[c0].index(block_name) + len(block_name+":")

        c1 = _getIndexOfBlock(contents[c0:], "end") + c0
        s1 = contents[c1].index("end")

        block_str = contents[c0][s0:] + \
                    ''.join(contents[c0+1:c1]) + \
                    contents[c1][:s1] + \
                    '\n'
    except IndexError as e:
        warnings.warn("\'%s\' block not found in .nestml file")
        block_str = ""

    return block_str


def getFunctionsString(contents):
    function_str = ""
    try:
        kk = 0
        while kk < len(contents):
            c0 = _getIndexOfBlock(contents[kk:], "function") + kk
            s0 = contents[c0].index("function")
            # print("\n",contents[kk:])

            c1 = _getIndexOfBlock(contents[c0:], "end") + c0
            s1 = contents[c1].index("end") + len("end")

            function_str += contents[c0][s0:] + \
                            ''.join(contents[c0+1:c1]) + \
                            contents[c1][:s1] + \
                            '\n'
            # move further, functions are assumed to at least occupy one line each
            kk = c1 + 1

    except (ValueError, IndexError) as e:
        # we reached the end of the NESTML file and don't have anymore functions
        # to check
        pass

    return function_str


def parseNestmlFile(f_name):
    with open(f_name, 'r') as file:
        contents = file.readlines()

    stripComments(contents)
    stripVComp(contents)

    blocks = copy.deepcopy(BLOCKS_EMPTY)
    for block_name in blocks:
        if block_name != "output" and block_name != "functions":
            blocks[block_name] += getBlockString(contents, block_name)
        elif block_name == "functions":
            blocks[block_name] += getFunctionsString(contents)

    return blocks


def writeNestmlBlocks(blocks, path_name, neuron_name, v_comp=0.,
                      write_blocks=['parameters', 'state', 'equations',
                                    'inputs', 'output', 'functions']):
    idx = blocks['state'].find("state:") + 6
    blocks['state'] = blocks['state'][:idx] + \
                      "\n    v_comp real = %.8f \n"%v_comp + \
                      blocks['state'][idx:]

    for block, blockstr in blocks.items():
        if block != 'functions' and block != 'output':
            blocks[block] = block + ":\n" + blockstr + 'end\n\n'
            # blocks[block] = blockstr + 'end\n\n'

    fname = os.path.join(path_name, neuron_name + ".nestml")

    file = open(fname, 'w')
    file.write('\nneuron %s:\n'%neuron_name)
    for blockstr in blocks.values():
        file.write(blockstr)
    file.write('\nend')
    file.close()

    return fname

if __name__ == "__main__":
    # parseNestmlFile("default_syns.nestml")
    writeNestmlBlocks(blocks_default_syns, "", "default_syns")
