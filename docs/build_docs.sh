rm quara.*.rst
cd ../
sphinx-apidoc -F -e -M -o docs/ quara/
cd docs/
make html
