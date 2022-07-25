from res.enkf import ForwardLoadContext

from ...libres_utils import ResTest


class ForwardLoadContextTest(ResTest):
    def test_create(self):
        ctx = ForwardLoadContext(report_step=1)
        self.assertEqual(1, ctx.getLoadStep())
