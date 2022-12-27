#!/bin/bash

JAVA_ARGS="-Xmx1g -classpath MOEAFramework-2.13-Demo.jar"
NUM_SAMPLES=1000
METHOD=latin

RANGES_FILENAME=LS98_IU_ranges.txt
OUTPUT_FILENAME=LS98_IU_all.txt
CSV_FILENAME=LS98_IU_Samples.csv
java ${JAVA_ARGS} org.moeaframework.analysis.sensitivity.SampleGenerator -m ${METHOD} -n ${NUM_SAMPLES} -p ${RANGES_FILENAME} -o ${OUTPUT_FILENAME}

sed 's/ /,/g' ${OUTPUT_FILENAME} > ${CSV_FILENAME}
