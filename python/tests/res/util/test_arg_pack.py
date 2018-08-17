from tests import ResTest
from res.util import ArgPack
from ecl.util.util import StringList


class ArgPackTest(ResTest):
    def test_create(self):
        arg = ArgPack()
        self.assertEqual(len(arg), 0)

        arg.append(StringList())
        self.assertEqual(len(arg), 1)

        arg.append(3.14)
        self.assertEqual(len(arg), 2)

        o = object()
        with self.assertRaises(TypeError):
            arg.append(o)

    def test_args(self):
        arg = ArgPack(1, 2, 3)
        self.assertEqual(len(arg), 3)

    def test_append_ptr(self):
        arg = ArgPack(StringList())
        self.assertEqual(len(arg), 1)
