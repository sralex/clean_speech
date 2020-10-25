from setuptools import setup, find_packages

setup(
    name='clean_speech',
    version='0.0.2	',
    packages=find_packages(),#['speech_enhance'],
    package_data={'clean_speech': ['*.json','*.h5']},
    include_package_data=True,
    scripts=['bin/clean_speech'],
    install_requires=[
    	"scipy>=1.5.2",
        "tensorflow==1.5.0",
        "keras==2.2.3",
        "soundfile>=0.10.3.post1"
      ],
    zip_safe=False
)
