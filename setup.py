from setuptools import setup

setup(
    name='torch_speech_dataloader',
    version='0.0.1',    
    description='A ready-to-use pytorch dataloader for audio classification, speech classification, speaker recognition, etc. with in-GPU augmentations',
    url='https://github.com/zabir-nabil/torch-speech-dataloader',
    author='Zabir Al Nazi',
    author_email='zabiralnazi@yahoo.com',
    license='MIT',
    packages=['torch_speech_dataloader'],
    install_requires=['torch_audiomentations>=0.11.0',
                      'numpy>=1.20.2',
                      'matplotlib==3.5.3',
                      'torch',
                      'torchaudio',
                      'scipy'                     
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research/Audio Processing',
        'License :: MIT',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)