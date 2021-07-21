# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 12:27:43 2021

@author: jon-f
"""

from setuptools import setup, find_packages

VERSION = '0.0.5' 
DESCRIPTION = 'Managing GEE data easily'
LONG_DESCRIPTION = 'A Python package for easily managing and downloading \
    Google Earth Engine data defined over geographical area. See https://github.com/triviajon/geearea \
        for detailed source code.'

setup(
        name="geearea", 
        version=VERSION,
        author="Jon Rosario",
        author_email="<jonf.rosario@gmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=['folium', 'earthengine-api', 'matplotlib', 'numpy'], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'google earth engine', 'gee', 'area'],
        classifiers= [
            "Development Status :: 4 - Beta",
            "Topic :: Utilities",
            "License :: OSI Approved :: MIT License",

        ]
)