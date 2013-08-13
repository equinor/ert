from unittest2 import TestLoader, TextTestRunner

if __name__ == '__main__':
    loader = TestLoader()
    tests = loader.discover('.')
    testRunner = TextTestRunner()
    testRunner.run(tests)