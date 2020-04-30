cd /Users/pcotton/github/homogenize/
rm /Users/pcotton/github/homogenize/dist/*
python setup.py sdist bdist_wheel
twine upload dist/*
