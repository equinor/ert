#!/prog/sdpsoft/python2.4/bin/python
import datetime
import ert
import ert.ecl as ecl

pos = (5354 , 9329 , 100)
grid     = ecl.EclGrid("data/eclipse/case/ECLIPSE.EGRID")
init     = ecl.EclFile("data/eclipse/case/ECLIPSE.INIT")
restart1 = ecl.EclFile.restart_block("data/eclipse/case/ECLIPSE.UNRST" , report_step = 10)
restart2 = ecl.EclFile.restart_block("data/eclipse/case/ECLIPSE.UNRST" , report_step = 40)



# Troll in Bergen
# pos      = (530991 , 6754822 , 342.785)
# grid     = ecl.EclGrid("/d/proj/bg/restroll2/restek2/TEG/simu_HM2009/BCUPD_HISTORYMATCH_JAN10_059.EGRID")
# init     = ecl.EclFile("/d/proj/bg/restroll2/restek2/TEG/simu_HM2009/BCUPD_HISTORYMATCH_JAN10_059.INIT")
# restart1 = ecl.EclFile.restart_block("/d/proj/bg/restroll2/restek2/TEG/simu_HM2009/BCUPD_HISTORYMATCH_JAN10_059.UNRST" , report_step = 10)
# restart2 = ecl.EclFile.restart_block("/d/proj/bg/restroll2/restek2/TEG/simu_HM2009/BCUPD_HISTORYMATCH_JAN10_059.UNRST" , report_step = 40)

if restart1 and restart2:
    deltaG = ecl.ecl_grav.deltag( pos , grid , init , restart1 , restart2 )

    print deltaG
else:
    print "Load failed"
