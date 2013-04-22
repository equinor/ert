import sys

PYTHONPATH = sys.argv[1]
sys.path.insert( 0 , PYTHONPATH )

print PYTHONPATH
print sys.path

test_module = __import__(sys.argv[2])
if len(sys.argv) > 3:
    test_module.ctest_run( sys.argv[3:] )
else:
    test_module.ctest_run( [ ] )
