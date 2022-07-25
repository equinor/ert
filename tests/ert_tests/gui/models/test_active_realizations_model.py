from res.config.rangestring import mask_to_rangestring

from ...ert_utils import ErtTest


class ActiveRealizationsModelTest(ErtTest):
    def testMaskToRangeConversion(self):
        cases = (
            ([0, 1, 0, 1, 1, 1, 0], "1, 3-5"),
            ([0, 1, 0, 1, 1, 1, 1], "1, 3-6"),
            ([0, 1, 0, 1, 0, 1, 0], "1, 3, 5"),
            ([1, 1, 0, 0, 1, 1, 1, 0, 1], "0-1, 4-6, 8"),
            ([1, 1, 1, 1, 1, 1, 1], "0-6"),
            ([0, 0, 0, 0, 0, 0, 0], ""),
            ([True, False, True, True], "0, 2-3"),
            ([], ""),
        )

        def nospaces(s):
            return "".join(s.split())

        for mask, expected in cases:
            rngstr = mask_to_rangestring(mask)
            self.assertEqual(
                nospaces(rngstr),
                nospaces(expected),
                msg=(
                    f"Mask {mask} was converted to {rngstr} which is different "
                    f"from the expected range {expected}"
                ),
            )
