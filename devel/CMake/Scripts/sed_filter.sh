#!/bin/sh
sed -e's/<TYPE>/$2/g' $1.c > $3/$2_vector.c
sed -e's/<TYPE>/$2/g' $1.h > $3/$2_vector.h
