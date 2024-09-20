-- standard operational settings:
-- producer BHP 395 bar
-- injector surface liquid rate target 97.5 SM3/D, BHP limit 420 bar

RPTSCHED
    FIP WELSPECS WELLS /

RPTRST
   BASIC=3 FREQ=1 /

WELSPECS
    'INJECT1' 'INJECT'   5    57  1* 'WATER' /
    'INJECT2' 'INJECT'   30   53  1* 'WATER' /
    'INJECT3' 'INJECT'   2    35  1* 'WATER' /
    'INJECT4' 'INJECT'   27   29  1* 'WATER' /
    'INJECT5' 'INJECT'   50   35  1* 'WATER' /
    'INJECT6' 'INJECT'   8    9   1* 'WATER' /
    'INJECT7' 'INJECT'   32   2   1* 'WATER' /
    'INJECT8' 'INJECT'   57   6   1* 'WATER' /
    'PROD1'   'PRODUC'   16   43  1* 'OIL' /
    'PROD2'   'PRODUC'   35   40  1* 'OIL' /
    'PROD3'   'PRODUC'   23   16  1* 'OIL' /
    'PROD4'   'PRODUC'   43   18  1* 'OIL' /
/

COMPDAT
    'INJECT1'    2*    1     7 'OPEN' 2*     0.2 	1*          0 /
    'INJECT2'    2*    1     7 'OPEN' 2*     0.2 	1*          0 /
    'INJECT3'    2*    1     7 'OPEN' 2*     0.2 	1*          0 /
    'INJECT4'    2*    1     7 'OPEN' 2*     0.2 	1*          0 /
    'INJECT5'    2*    1     7 'OPEN' 2*     0.2 	1*          0 /
    'INJECT6'    2*    1     7 'OPEN' 2*     0.2 	1*          0 /
    'INJECT7'    2*    1     7 'OPEN' 2*     0.2 	1*          0 /
    'INJECT8'    2*    1     7 'OPEN' 2*     0.2 	1*          0 /
    'PROD1'      2*    1     7 'OPEN' 2*     0.2 	1*          0 /
    'PROD2'      2*    1     7 'OPEN' 2*     0.2 	1*          0 /
    'PROD3'      2*    1     7 'OPEN' 2*     0.2 	1*          0 /
    'PROD4'      2*    1     7 'OPEN' 2*     0.2	1*          0 /
/

WCONPROD
    'PROD1' 'SHUT' 'BHP' 5*  395 /
    'PROD3' 'SHUT' 'BHP' 5*  395 /
    'PROD2' 'SHUT' 'BHP' 5*  395 /
    'PROD4' 'SHUT' 'BHP' 5*  395 /
/

WCONINJE
    'INJECT1'	'WATER'	'SHUT'	'GRUP'	79.5 1* 420 /
    'INJECT2'	'WATER'	'SHUT'	'GRUP'	79.5 1* 420 /
    'INJECT3'	'WATER'	'SHUT'	'GRUP'	79.5 1* 420 /
    'INJECT4'	'WATER'	'SHUT'	'GRUP'	79.5 1* 420 /
    'INJECT5'	'WATER'	'SHUT'	'GRUP'	79.5 1* 420 /
    'INJECT6'	'WATER'	'SHUT'	'GRUP'	79.5 1* 420 /
    'INJECT7'	'WATER'	'SHUT'	'GRUP'	79.5 1* 420 /
    'INJECT8'	'WATER'	'SHUT'	'GRUP'	79.5 1* 420 /
/

TUNING
0.1 30 /
/
12 1 250 1* 25 /

GRUPTREE
 'PRODUC' 'FIELD' /
 'INJECT' 'FIELD' /
/

GCONINJE
 'INJECT' 'WATER' 'VREP' 30000 2* 1 4* 'PRODUC' /
/
