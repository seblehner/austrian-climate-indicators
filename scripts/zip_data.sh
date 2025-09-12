#!/bin/sh

# todo
DATDIR="/perm/klifol/slehner/"
PREFIX="austrian_climate_indicators"
echo "Zip climate indicator folders in $DATDIR$PREFIX*"
cd $DATDIR
mkdir -p zip
for SUFFIX in "" "_areamean" "_clim_1961_1990" "_clim_1991_2020" "_plots" "_significance"
do
    ZIPDIR=$PREFIX$SUFFIX
    echo "Zipping $ZIPDIR into $ZIPDIR.zip"
    zip -r -Z bzip2 zip/$ZIPDIR.zip $ZIPDIR
done
echo "Finished zipping"
