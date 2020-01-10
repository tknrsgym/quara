rm quara*.rst
cd ../
sphinx-apidoc -F -o docs/ quara/
cd docs/
make html
