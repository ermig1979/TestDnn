ECHO="0"
HT="1"
BUILD_DIR=build

if [ ${ECHO} == "1" ]; then echo "Build TestDnn in '${BUILD_DIR}':"; fi

if [ ! -d $BUILD_DIR ]; then mkdir $BUILD_DIR; fi

cd $BUILD_DIR

cmake ../prj/cmake -DTOOLCHAIN="/usr/bin/c++" -DCMAKE_BUILD_TYPE=Release
if [ $? -ne 0 ] ; then 	exit; fi

if [ ${HT} == "1" ]; then make "-j$(nproc)"; else make "-j$(grep "^core id" /proc/cpuinfo | sort -u | wc -l)"; fi
if [ $? -ne 0 ] ; then 	exit; fi
