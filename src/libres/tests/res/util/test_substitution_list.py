from tests import ResTest
from res.util.substitution_list import SubstitutionList


class SubstitutionListTest(ResTest):
    def test_substitution_list(self):
        subst_list = SubstitutionList()

        subst_list.addItem("Key", "Value", "Doc String")

        self.assertEqual(len(subst_list), 1)

        with self.assertRaises(KeyError):
            item = subst_list[2]

        with self.assertRaises(KeyError):
            item = subst_list["NoSuchKey"]

        with self.assertRaises(KeyError):
            item = subst_list.doc("NoSuchKey")

        self.assertTrue("Key" in subst_list)
        self.assertEqual(subst_list["Key"], "Value")
        self.assertEqual(subst_list.doc("Key"), "Doc String")

        subst_list.addItem("Key2", "Value2", "Doc String2")
        self.assertEqual(len(subst_list), 2)

        keys = subst_list.keys()
        self.assertEqual(keys[0], "Key")
        self.assertEqual(keys[1], "Key2")

        self.assertIn("Key", str(subst_list))
        self.assertIn("SubstitutionList", repr(subst_list))
        self.assertIn("2", repr(subst_list))

        self.assertEqual(1729, subst_list.get("nosuchkey", 1729))
        self.assertIsNone(subst_list.get("nosuchkey"))
        self.assertIsNone(subst_list.get(513))
        for key in ("Key", "Key2"):
            self.assertEqual(subst_list[key], subst_list.get(key))
