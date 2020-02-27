from distutils.core import setup

from setuptools.command.install import install


class CustomInstallCommand(install):
    """Customized setuptools install command"""

    def run(self):
        install.run(self)


# current location: neat/tools/simtools/neuron/

setup(
    name='neatneuron',
    packages=['neatneuron'],  # to make sure the morphologies folder is copied to the installation directory
    package_data={'neatneuron': ['morph/*.swc', 'mech/*.mod', 'compilemechs.sh']},
    cmdclass={'install': CustomInstallCommand},
    include_package_data=True,
    author='Willem Wybo',
)
