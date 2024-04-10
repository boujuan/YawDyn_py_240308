#!/bin/bash
#check first or last lines of file
#head VA_ActivePowerReferenceDemandCommand_Param1_N3_3.ASC
#tail VA_ActivePowerReferenceDemandCommand_Param1_N3_3.ASC

#delete first two lines of all files in directory using sed with inplace editing

sed -i '1,2d' *.ASC
