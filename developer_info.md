
#### Necessary steps for making a new release
* Check citation.cff using general DOI for all version (option: create file via 'cffinit')
* Create .zenodo.json file from CITATION.cff (using cffconvert)  
```cffconvert --validate```  
```cffconvert --ignore-suspect-keys --outputformat zenodo --outfile .zenodo.json```
* Set new version number in mcfly/_version.py
* Edit Changelog (based on commits in https://github.com/NLeSC/mcfly/compare/v1.0.1...master)
* Create Github release
* Upload to pypi:  
```python setup.py sdist bdist_wheel```  
```python -m twine upload --repository-url https://upload.pypi.org/legacy/ dist/*```  
(or ```python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*``` to test first)
* Check doi on zenodo
* If the visualization has changed, deploy it to github pages:
```
git subtree push --prefix html origin gh-pages
```
