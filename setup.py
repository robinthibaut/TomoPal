from setuptools import setup, find_packages

my_pckg = find_packages()
print(my_pckg)

setup(
    name='tomopal',
    version='1.0.19',
    packages=my_pckg,
    include_package_data=True,
    url='https://github.com/robinthibaut/tomopal',
    license='MIT',
    author='Robin Thibaut',
    author_email='robin.thibaut@UGent.be',
    description='Your electrical resistivity tomography companion !',
    long_description='Your electrical resistivity tomography companion !',
    install_requires=['numpy', 'rasterio', 'vtk', 'geographiclib', 'scipy', 'pandas', 'matplotlib'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
