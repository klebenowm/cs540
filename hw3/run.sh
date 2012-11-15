#!/bin/bash
javac *.java;
T="$(date +%s)"
java HW3 sms.txt sms.txt > tmp.txt;
T="$(($(date +%s)-T))"
printf "Total Texts: ";
grep "AM" tmp.txt | wc -l;
printf "False Positives: ";
grep "SPAM HAM" tmp.txt | wc -l;
printf "False Negatives: ";
grep "HAM SPAM" tmp.txt | wc -l;
rm -f *.class;
rm -f tmp.txt;
echo "Time: ${T}sec"
