from setuptools import setup

setup(
    name='polyroots',
    version='0.0.1',
    packages=['tests', 'methods'],
    url='https://github.com/lxnk/polyroots',
    license='MIT',
    author='Alex Kashuba',
    author_email='o.kashuba@gmail.com',
    description='A summary of methods for finding polynomial roots',
    long_description='A summary of methods for finding polynomial roots: Housholder, Aberth, Bairstow, Durand, Graeffe, Laguerre, Vincent',
    classifiers=['Development Status :: 2 - Pre-Alpha',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 3.7',
                 'Topic :: Solving algorithms :: Numerics'],
    keywords='numerics roots polynomial'
)
