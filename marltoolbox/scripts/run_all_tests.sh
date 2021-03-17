# This should be similar to the tests in
# marltoolbox/.github/workflows/weekly_tests.yml

# Lint
# stop the build if there are Python syntax errors or undefined names
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
# exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Pytest
python -m pytest tests --cov

# Test notebooks
jupyter nbconvert ./marltoolbox/examples/Tutorial_Basics_How_to_use_the_toolbox.ipynb --to script
python ./marltoolbox/examples/Tutorial_Basics_How_to_use_the_toolbox.py
jupyter nbconvert ./marltoolbox/examples/Tutorial_Evaluations_Level_1_best_response_and_self_play_and_cross_play.ipynb --to script
python ./marltoolbox/examples/Tutorial_Evaluations_Level_1_best_response_and_self_play_and_cross_play.py

# Long training runs
python -m pytest ./tests/marltoolbox/examples/manual_test_end_to_end.py