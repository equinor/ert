import os

try:
    from unittest2 import TestLoader, TextTestRunner
except ImportError:
    from unittest import TestLoader, TextTestRunner


class ErtTestRunner(object):

    @staticmethod
    def runTestsInDirectory(path=".", recursive=True):
        if recursive:
            for (root, dirnames, filenames) in os.walk( path ):
                for directory in dirnames:
                    ErtTestRunner.runTestsInDirectory(os.path.join(root, directory), recursive )


        loader = TestLoader()
        tests = loader.discover(path)

        if tests.countTestCases() > 0:
            print("Running %d tests in %s" % (tests.countTestCases(), path))

        testRunner = TextTestRunner(verbosity=3)
        testRunner.run(tests)


    @staticmethod
    def runTestsInClass(classpath):
        klass = ErtTestRunner.importClass(classpath)
        loader = TestLoader()
        tests = loader.loadTestsFromTestCase(klass)
        testRunner = TextTestRunner()
        testRunner.run(tests)

    @staticmethod
    def importClass(classpath):
        dot = classpath.rfind(".")
        class_name = classpath[dot + 1:len(classpath)]
        m = __import__(classpath[0:dot], globals(), locals(), [class_name])
        return getattr(m, class_name)

    @staticmethod
    def getTestsFromTestClass(test_class_path, argv=None):
        klass = ErtTestRunner.importClass(test_class_path)
        klass.argv = argv
        loader = TestLoader()
        return loader.loadTestsFromTestCase(klass)



