# Contributing

## <a name="issue"></a> Filing an Issue

This is the way to report issues
and also to request features.

1. Go to 
   [Graph-mol's GitHub Issues](https://github.com/yashasvi-ranawat/Graph-mol/issues).
2. Look through open and closed issues to see if there has already been a
   similar bug report or feature request.
   [GitHub's search](https://github.com/yashasvi-ranawat/Graph-mol/search)
   feature may aid you as well.
3. If your issue doesn't appear to be a duplicate,
   [file an issue](https://github.com/yashasvi-ranawat/Graph-mol/issues/new).
4. Please be as descriptive as possible!

## Pull Request

1. Fork the repository on [GitHub](https://github.com/yashasvi-ranawat/Graph-mol).
2. Create a new branch for your work.
3. Install development python dependencies, by running 
   [pipenv](https://pipenv.pypa.io/en/latest/) in the top-level directory with
   `Pipfile`:

   ```shell
   $ pipenv sync --dev
   $ pipenv shell
   ```

4. Make your changes (see below).
5. Send a GitHub Pull Request to the ``master`` branch of ``Graph-mol``.

Step 4 is different depending on if you are contributing to the code base or
documentation.

### Code

1. Run pytest using
   ```shell
   make test
   ```
   If any tests fail, and you are unable to diagnose the reason, please refer
   to [Filing an Issue](#issue).
2. Complete your patch and write tests to verify your fix or feature is working.
   Please try to reduce the size and scope of your patch to make the review
   process go smoothly.
3. Run the tests again and make any necessary changes to ensure all tests pass.
4. Run [black](https://black.readthedocs.io/en/stable/index.html) to ensure correct
   linting of the python code, in the top-level directory as:
   ```shell
   $ make black
   ```

### Documentation

1. Enter the [docs](https://github.com/yashasvi-ranawat/Graph-mol/tree/master/docs) directory.
2. Make your changes to any files.
3. Run ``make clean && make html``. This will generate html files in a new
   ``_build/html/`` directory.
4. Open the generated pages and make any necessary changes to the ``.rst``
   files until the documentation looks properly formatted.

## TODO

1. Add docs
2. Implement multiple model handling
3. Add [TokenGT](https://arxiv.org/pdf/2207.02505.pdf)
4. Implement half precision 
