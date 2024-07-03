from setuptools import setup, find_packages
# from codecs import open
# from os import path

__author__ = 'Salvatore Citraro'
__license__ = "BSD-2-Clause"
__email__ = "salvatore.citraro@isti.cnr.it"

# here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
# with open(path.join(here, 'README.md'), encoding='utf-8') as f:
#    long_description = f.read()


setup(name='SpreadPy',
      version='1.0.0',
      license='BSD-Clause-2',
      description='SpreadPy: Semantic Processing Simulation in Python',
      url='https://github.com/dsalvaz/SpreadPy/',
      author=['Salvatore Citraro'],
      author_email='salvatore.citraro@isti.cnr.it',
      classifiers=[
          # How mature is this project? Common values are
          #   3 - Alpha
          #   4 - Beta
          #   5 - Production/Stable
          'Development Status :: 4 - Beta',

          # Indicate who your project is intended for
          'Intended Audience :: Developers',
          'Topic :: Software Development :: Build Tools',

          # Pick your license as you wish (should match "license" above)
          'License :: OSI Approved :: BSD License',

          "Operating System :: OS Independent",

          # Specify the Python versions you support here. In particular, ensure
          # that you indicate whether you support Python 2, Python 3 or both.
          'Programming Language :: Python :: 3'
      ],
      keywords='',
      install_requires=['six', 'numpy', 'tqdm', 'networkx', 'matplotlib', 'pandas', 'nltk', 'conformity', 'scipy'],
      packages=find_packages(exclude=["*.test", "*.test.*", "test.*", "test", "SpreadPy.test", "SpreadPy.test.*"]),
      )