#!/bin/sh
cmd="s/<TYPE>/$2/g"
sed -e${cmd} $1.c > $3/$2_vector.c
sed -e${cmd} $1.c > $3/$2_vector.h
