"""
writes nestml for all channels defined in ``neat.channels.channelcollection``
"""
import os

from channelcollection import channelcollection

print('--> Writing mod files')

blocks = {'parameters': '\nparameters:\n',
          'state': '\nstate:\n    v_comp real\n',
          'equations': '\nequations:\n',
          'functions': '\n'
          }

for name, channel_class in list(channelcollection.__dict__.items()):

    if isinstance(channel_class, type) and name != 'IonChannel' and name != '_func':
        chan = channel_class()
        blocks_ = chan.writeNestmlBlocks()
        del chan

        for block, blockstr in blocks_.items():
            blocks[block] += blockstr

for block, blockstr in blocks.items():
    if block != 'functions':
        blocks[block] = blockstr + 'end\n\n'

fname = os.path.join(os.path.dirname(__file__), '../tools/simtools/nest/default.nestml')

file = open(fname, 'w')
file.write('\nneuron default:\n')
for blockstr in blocks.values():
    file.write(blockstr)
file.write('\nend')
file.close()

