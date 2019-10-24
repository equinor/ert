import collections
import json
import os
import unittest


import ert3


from tests.utils import tmpdir


class TestResourceStorage(unittest.TestCase):

    @tmpdir()
    def test_initalize(self):
        storage_name = "my_storage"
        ert3.resource_storage.initialize(storage_name)
        self.assertEqual(
            (),
            ert3.resource_storage.load_resource_names(storage_name),
        )

    @tmpdir()
    def test_initalize_twice(self):
        storage_name = "my_storage"
        ert3.resource_storage.initialize(storage_name)
        with self.assertRaises(ValueError):
            ert3.resource_storage.initialize(storage_name)

    @tmpdir()
    def test_add_resource(self):
        storage_name = "my_storage"
        ert3.resource_storage.initialize(storage_name)

        resource_name = "100x"
        resource = 100*"x"
        ert3.resource_storage.add_resource(resource_name, resource, storage_name)
        self.assertEqual(
            (resource_name,),
            ert3.resource_storage.load_resource_names(storage_name),
        )
        self.assertEqual(
            resource,
            ert3.resource_storage.load_resource(resource_name, storage_name),
        )

    @tmpdir()
    def test_add_multiple_resources(self):
        storage_name = "my_storage"
        ert3.resource_storage.initialize(storage_name)

        for i in range(100):
            resource_name = "{}x".format(i)
            resource = i*"x"
            ert3.resource_storage.add_resource(resource_name, resource, storage_name)

        self.assertEqual(
            tuple("{}x".format(i) for i in range(100)),
            ert3.resource_storage.load_resource_names(storage_name),
        )

        for i in range(100):
            resource_name = "{}x".format(i)
            resource = i*"x"
            self.assertEqual(
                resource,
                ert3.resource_storage.load_resource(resource_name, storage_name),
            )

    @tmpdir()
    def test_add_json_resource(self):
        storage_name = "my_storage"
        ert3.resource_storage.initialize(storage_name)

        resource_name = "100x"
        resource = {"name": resource_name, "elems": list(range(100))}
        ert3.resource_storage.add_json_resource(resource_name, resource, storage_name)
        self.assertEqual(
            (resource_name,),
            ert3.resource_storage.load_resource_names(storage_name),
        )
        self.assertEqual(
            resource,
            ert3.resource_storage.load_json_resource(resource_name, storage_name),
        )

    @tmpdir()
    def test_add_json_resources(self):
        storage_name = "my_storage"
        ert3.resource_storage.initialize(storage_name)

        for i in range(100):
            resource_name = "{}x".format(i)
            resource = i*"x"
            ert3.resource_storage.add_json_resource(resource_name, resource, storage_name)

        self.assertEqual(
            tuple("{}x".format(i) for i in range(100)),
            ert3.resource_storage.load_resource_names(storage_name),
        )

        for i in range(100):
            resource_name = "{}x".format(i)
            resource = i*"x"
            self.assertEqual(
                resource,
                ert3.resource_storage.load_json_resource(resource_name, storage_name),
            )

    @tmpdir()
    def test_add_multiple_storage(self):
        storage1 = "storage_one"
        storage2 = "storage_two"

        ert3.resource_storage.initialize(storage1)
        ert3.resource_storage.initialize(storage2)

        resource_name = "100x"
        resource = 100*"x"

        ert3.resource_storage.add_resource(resource_name, resource+"1", storage1)
        ert3.resource_storage.add_resource(resource_name, resource+"2", storage2)

        self.assertEqual(
            (resource_name,),
            ert3.resource_storage.load_resource_names(storage1),
        )
        self.assertEqual(
            (resource_name,),
            ert3.resource_storage.load_resource_names(storage2),
        )

        self.assertEqual(
            resource+"1",
            ert3.resource_storage.load_resource(resource_name, storage1),
        )
        self.assertEqual(
            resource+"2",
            ert3.resource_storage.load_resource(resource_name, storage2),
        )

    @tmpdir()
    def test_invalid_resource_name(self):
        storage_name = "my_storage"
        ert3.resource_storage.initialize(storage_name)

        with self.assertRaises(ValueError):
            ert3.resource_storage.add_resource(
                "../../my_resource",
                "xxx",
                storage_name,
            )
