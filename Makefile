.PHONY: r-doc r-build r-test r-check r-all py-build py-test py-all docs all

r-doc:
	Rscript -e "devtools::document()"

r-sync-headers:
	mkdir -p inst/include
	cp include/*.hpp inst/include/

r-build: r-sync-headers r-doc
	R CMD INSTALL .

r-test:
	Rscript -e "tinytest::test_package('robustrolling')"

r-check: r-sync-headers r-doc
	R CMD build .
	R CMD check --as-cran robustrolling_*.tar.gz

r-all: r-build r-test

py-build:
	.venv/bin/pip install -e py_package/ --no-cache-dir

py-test:
	.venv/bin/pytest py_package/tests/ -v --tb=short

py-all: py-build py-test

docs:
	py_package/venv/bin/python -m sphinx -b html docs/python docs/_build/python
	@echo "Docs built"

all: r-all py-all
