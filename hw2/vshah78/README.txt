README.txt

I directly modified files in the source code to cater to my dataset and this project

To run the code:

PART 1:

from the vshah78-code directory
cd ABAGAIL/jython
export CLASSPATH=../jython

the following commands run their corresponding algorithms the way I did

jython abalone_test.py BP
jython abalone_test.py RHC
jython abalone_test.py SA
jython abalone_test.py GA

and produce corresponding csv files with their data
all my data/graphs/calculations are visible at https://docs.google.com/spreadsheets/d/1c7mr-mXj7cr4W85jG3JqZRn39xxSaNNclnkBT9ymLiY/edit?usp=sharing


PART 2:

from the vshah78-code directory
cd ABAGAIL/

the following commands run the corresponding problems

java -cp ABAGAIL.jar opt.test.KnapsackTest
java -cp ABAGAIL.jar opt.test.TravelingSalesmanTest
java -cp ABAGAIL.jar opt.test.FlipFlopTest
