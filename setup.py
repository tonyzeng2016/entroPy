#encoding=utf-8
u'''
Created on 2016-09-18

@author: zengxiaolong2015@163.com
'''
from distutils.core import setup, Extension  
#from setuptools import setup,  Extension
import glob

setup (name = 'entroPy',
       version = '0.1.0',
       description='maximum entropy (MaxEnt) classifier',
       author='xiaolong,zeng',
       author_email='zengxiaolong2015@163.com',
       ext_modules =[Extension('_model',sources=glob.glob('c/*.c'))],
       ext_package='entroPy',
       packages=['entroPy'],
       package_dir={'entroPy':'py'},
license = 'GNU GPL version 3',
       #package_data={'':['*']},
 classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Programming Language :: C',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        ],
    zip_safe=False
       ) 
