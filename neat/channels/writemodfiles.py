"""
writes mod files for all channels defined in ``neat.channels.channelcollection``
"""
import os

from . import channelcollection

print('--> Writing mod files')
for name, channel_class in list(channelcollection.__dict__.items()):
    if isinstance(channel_class, type) and name != 'IonChannel' and name != '_func':
        chan = channel_class()
        chan.write_mod_file(os.path.join(os.path.dirname(__file__),
                                       '../simulations/neuron/mech'))
        del chan
