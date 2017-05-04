from ecl.test import ExtendedTestCase
from res.enkf import ForwardLoadContext

class ForwardLoadContextTest(ExtendedTestCase):

    def test_create(self):
        ctx = ForwardLoadContext( report_step = 1 )
        self.assertEqual( 1 , ctx.getLoadStep( ) )
        
