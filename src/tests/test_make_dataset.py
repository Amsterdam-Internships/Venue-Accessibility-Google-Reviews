"""
Example of a (slighly pointless) test.
    It will be run every time you push thanks to .github/workflows/test_workflow.yml
If a test fails you will get an AssertionError and output would be something like:

======================================= test session starts =======================================
platform linux -- Python 3.8.5, pytest-6.2.3, py-1.10.0, pluggy-0.13.1
rootdir: /home/iva/projects/gemeente/InternshipAmsterdam/InternshipAmsterdamGeneral
collected 2 items

tests/example_tests.py .F                                                                   [100%]

============================================ FAILURES =============================================
_______________________________________ test_that_will_fail _______________________________________

    def test_that_will_fail():
>       assert 0 == 1
E       assert 0 == 1

tests/example_tests.py:57: AssertionError
===================================== short test summary info =====================================
FAILED tests/example_tests.py::test_that_will_fail - assert 0 == 1
=================================== 1 failed, 1 passed in 0.11s ===================================
"""

import numpy as np
import pandas as pd
import os
import sys
import pytest
from ..data.make_dataset import load_data, make_trainset, make_testset

@pytest.fixture(scope="session")
def test_data_paths():
    # check the file path
    train_path = os.environ.get('RAW_TRAIN_DATA_PATH')
    
    test_path = os.environ.get('RAW_TEST_DATA_PATH')

    # check that file is saved 
    processed_path = os.environ.get('PROCESSED_DATA')
    
    # Test that processed data file was created
    assert os.path.exists(train_path) == True
    assert os.path.exists(test_path) == True
    assert os.path.exists(processed_path) == True
