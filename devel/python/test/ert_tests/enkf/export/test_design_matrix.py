from ert.enkf.export import DesignMatrixReader
from ert.test import ExtendedTestCase


class DesignMatrixTest(ExtendedTestCase):

    def setUp(self):
        self.design_matrix_file = self.createTestPath("Statoil/ert-statoil/spotfire/DesignMatrix.txt")
        self.design_matrix_file_2 = self.createTestPath("Statoil/ert-statoil/spotfire/DesignMatrix2.txt")

    def test_read_design_matrix(self):
        dm = DesignMatrixReader.loadDesignMatrix(self.design_matrix_file)

        self.assertEqual(dm["CORR_SEIS_HEIMDAL"][0], 0.8)
        self.assertEqual(dm["VOL_FRAC_HEIMDAL"][0], 0.08)
        self.assertEqual(dm["AZIM_IND_HEIMDAL"][0], 125)
        self.assertEqual(dm["VARIO_PARAL_HEIMDAL"][0], 1000)
        self.assertEqual(dm["VARIO_NORM_HEIMDAL"][0], 500)
        self.assertEqual(dm["VARIO_VERT_HEIMDAL"][0], 25)
        self.assertEqual(dm["SEIS_COND_HEIMDAL"][0], "ON")

        self.assertEqual(dm["CORR_SEIS_HEIMDAL"][1], 0.8)
        self.assertEqual(dm["VOL_FRAC_HEIMDAL"][1], 0.15)
        self.assertEqual(dm["AZIM_IND_HEIMDAL"][1], 125)
        self.assertEqual(dm["VARIO_PARAL_HEIMDAL"][1], 2000)
        self.assertEqual(dm["VARIO_NORM_HEIMDAL"][1], 1000)
        self.assertEqual(dm["VARIO_VERT_HEIMDAL"][1], 25)
        self.assertEqual(dm["SEIS_COND_HEIMDAL"][1], "ON")

        self.assertEqual(dm["CORR_SEIS_HEIMDAL"][2], 0.8)
        self.assertEqual(dm["VOL_FRAC_HEIMDAL"][2], 0.20)
        self.assertEqual(dm["AZIM_IND_HEIMDAL"][2], 125)
        self.assertEqual(dm["VARIO_PARAL_HEIMDAL"][2], 4000)
        self.assertEqual(dm["VARIO_NORM_HEIMDAL"][2], 2000)
        self.assertEqual(dm["VARIO_VERT_HEIMDAL"][2], 25)
        self.assertEqual(dm["SEIS_COND_HEIMDAL"][2], "ON")


    def test_read_design_matrix_2(self):
        dm = DesignMatrixReader.loadDesignMatrix(self.design_matrix_file_2)

        self.assertEqual(dm["CORR_SEIS_HEIMDAL"][0], 0.8)
        self.assertEqual(dm["VOL_FRAC_HEIMDAL"][0], 0.08)
        self.assertEqual(dm["AZIM_IND_HEIMDAL"][0], 125)
        self.assertEqual(dm["VARIO_PARAL_HEIMDAL"][0], 1000)
        self.assertEqual(dm["VARIO_NORM_HEIMDAL"][0], 500)
        self.assertEqual(dm["VARIO_VERT_HEIMDAL"][0], 25)
        self.assertEqual(dm["SEIS_COND_HEIMDAL"][0], "ON")

        self.assertEqual(dm["CORR_SEIS_HEIMDAL"][1], 0.8)
        self.assertEqual(dm["VOL_FRAC_HEIMDAL"][1], 0.15)
        self.assertEqual(dm["AZIM_IND_HEIMDAL"][1], 125)
        self.assertEqual(dm["VARIO_PARAL_HEIMDAL"][1], 2000)
        self.assertEqual(dm["VARIO_NORM_HEIMDAL"][1], 1000)
        self.assertEqual(dm["VARIO_VERT_HEIMDAL"][1], 25)
        self.assertEqual(dm["SEIS_COND_HEIMDAL"][1], "ON")

        self.assertEqual(dm["CORR_SEIS_HEIMDAL"][2], 0.8)
        self.assertEqual(dm["VOL_FRAC_HEIMDAL"][2], 0.20)
        self.assertEqual(dm["AZIM_IND_HEIMDAL"][2], 125)
        self.assertEqual(dm["VARIO_PARAL_HEIMDAL"][2], 4000)
        self.assertEqual(dm["VARIO_NORM_HEIMDAL"][2], 2000)
        self.assertEqual(dm["VARIO_VERT_HEIMDAL"][2], 25)
        self.assertEqual(dm["SEIS_COND_HEIMDAL"][2], "ON")


        with self.assertRaises(KeyError):
            self.assertEqual(dm["CORR_SEIS_HEIMDAL"][3], 0.8)

        self.assertEqual(dm["CORR_SEIS_HEIMDAL"][4], 0.8)
        self.assertEqual(dm["VOL_FRAC_HEIMDAL"][4], 0.30)
        self.assertEqual(dm["AZIM_IND_HEIMDAL"][4], 125)
        self.assertEqual(dm["VARIO_PARAL_HEIMDAL"][4], 16000)
        self.assertEqual(dm["VARIO_NORM_HEIMDAL"][4], 8000)
        self.assertEqual(dm["VARIO_VERT_HEIMDAL"][4], 25)
        self.assertEqual(dm["SEIS_COND_HEIMDAL"][4], "ON")
