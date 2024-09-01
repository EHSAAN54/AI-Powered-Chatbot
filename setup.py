from setuptools import setup, find_packages

setup(
    name='AI-Powered-Chatbot',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'Flask==2.0.1',
        'TensorFlow==2.7.0',
        'spaCy==3.2.1',
        'nltk==3.6.3',
    ],
)
