#!/usr/bin/env bash

SS3_VER=$(curl -s https://api.github.com/repos/adobe-fonts/source-sans/releases/latest | grep -oP '"tag_name": "\K(.*)(?=")')
MONO_VER="2.042R-u%2F1.062R-i%2F1.026R-vf/OTF-source-code-pro-2.042R-u_1.062R-i.zip"

mkdir -p /usr/share/fonts/source-sans
ZIPFILE=OTF-source-sans-${SS3_VER}.zip
wget https://github.com/adobe-fonts/source-sans/releases/download/${SS3_VER}/${ZIPFILE}
unzip -j ${ZIPFILE} OTF/* -d /usr/share/fonts/source-sans/
rm -rf ${ZIPFILE}

mkdir -p /usr/share/fonts/source-code-pro/
wget https://github.com/adobe-fonts/source-code-pro/releases/download/${MONO_VER} -O source-code-pro.zip
unzip -j source-code-pro.zip OTF/* -d /usr/share/fonts/source-code-pro/
rm -rf source-code-pro.zip

fc-cache -f -v
