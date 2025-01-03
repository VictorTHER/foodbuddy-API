from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='foodbuddy',
      version="0.0.1",
      description="FoodBuddy Model (api_pred)",
      license="Victor THERY",
      author="Victor THERY",
      author_email="victorthery@orange.fr",
      url="https://github.com/VictorTHER/foodbuddy",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
