from ert_gui.models import ErtConnector
from ert_gui.models.mixins import BasicModelMixin
from ert_tests import ExtendedTestCase


class EmptyModel(ErtConnector, BasicModelMixin):
    def initialize(self):
        pass


class TestModel(ErtConnector, BasicModelMixin):

    def __init__(self):
        self.observable().addEvent("event")

    def initialize(self):
        self.value = None
        self.observable().notify("event")

    def getValue(self):
        return self.value

    def setValue(self, value):
        self.value = value
        self.observable().notify("event")




class ModelTest(ExtendedTestCase):

    def setUp(self):
        model = TestModel()
        model.setValue(None)

        ErtConnector.setErt(None)

    def test_abstract_model(self):
        model = EmptyModel()

        with self.assertRaises(NotImplementedError):
            model.getValue()

        # with self.assertRaises(NotImplementedError):
        #     model.getCollection()

        with self.assertRaises(NotImplementedError):
            model.setValue("Error")

        # with self.assertRaises(NotImplementedError):
        #     model.createValue("Error")
        #
        # with self.assertRaises(NotImplementedError):
        #     model.removeValue("Error")


    def test_observer(self):
        model = TestModel()

        def observe():
            self.assertEqual(model.getValue(), 2)

        model.observable().attach("event", observe)

        model.setValue(2)

        model.observable().detach("event", observe)

        model.setValue(None)

        self.assertEqual(model.getValue(), None)


    def test_initializer(self):
        model = TestModel()

        def init():
            if model.getValue() is None:
                raise ValueError("Init")
            else:
                self.assertEqual(model.getValue(), "Correct")

        model.observable().attach("event", init)

        with self.assertRaises(ValueError):
            ErtConnector.setErt("ERT")

        self.assertEqual(model.ert(), "ERT")

        model.setValue("Correct")

        model.observable().detach("event", init)

        model.setValue(None)


    def test_observability(self):
        model = EmptyModel()
        model1 = TestModel()
        model2 = TestModel()

        self.assertIsNone(ErtConnector())
        self.assertIsNotNone(model)
        self.assertIsNotNone(model1)
        self.assertIsNotNone(model2)

        self.assertNotEqual(model, model1)
        self.assertEqual(model1, model2)





