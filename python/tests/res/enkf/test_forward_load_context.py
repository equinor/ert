from tests import ResTest
from res.enkf import ForwardLoadContext


class ForwardLoadContextTest(ResTest):
    def test_create(self):
        ctx = ForwardLoadContext(report_step=1)
        self.assertEqual(1, ctx.getLoadStep())
