===========
Limitations
===========

errors in the Choi matrix entries of the result of qpt
------------------------------------------------------

Despite using the same OS and Matlab versions, we have found that the Choi matrix entries of the result of qpt may have about \1.5 x 10 :sup:`-7`\  absolute error depending on the machine.
This happens in our environment and also has been reported by Quara users.
We assume that the cause is Matlab or a library related to Matlab.

This happens in following functions:

- quara.protocol.simple_qpt.execute
- quara.protocol.simple_qpt.execute_from_csv

