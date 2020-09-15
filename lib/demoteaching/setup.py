from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='demoteaching',
      version='0.2',
      description='Teaching by demonstration package.',
      url='',
      author="Mark Ho",
      author_email='mark.ho.cs@gmail.com',
      license='MIT',
      packages=['demoteaching'],
      install_requires=[
          'markdown',
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)