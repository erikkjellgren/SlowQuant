
General Considerations
======================

Inside the code are four different objects that ties the different moduels together. The basisset obejct "basis", the inputfile object "input", the settingsfile object "settings" and the results object "results".

Basisset object
---------------

The basisset object is a list of list with the following structure:

.. math::
   \left[\begin{array}{ccccccc}Idx & x & y & z & \#CONTR & CONTR & Atomidx\end{array}\right]

Idx is the AO index. Atomidx, is the index of the associated atom. \#CONTR contains the number of primitive functions in the contracted.

CONTR contains all the information about the primitive functions and have the form:

.. math::
   \left[\begin{array}{cccccc}N & \zeta & c & l & m & n\end{array}\right]

Inputfile obejct
----------------

The inputfile obejct is a numpy array containing the following informations:

.. math::
   \left[\begin{array}{cccc}\#electrons & None & None & None\\Atom_{1}\,nr. & Atom_{1}\,x & Atom_{1}\,y & Atom_{1}\,z\\Atom_{2}\,nr. & Atom_{2}\,x & Atom_{2}\,y & Atom_{2}\,z\\... & ... & ... & ...\\Atom_{i}\,nr. & Atom_{i}\,x & Atom_{i}\,y & Atom_{i}\,z\end{array}\right]

Settings obejct
---------------

The settings are parsed around in the code as a dictionary.

Results object
--------------

The results are parsed around in the code as a dictionary.

Integral storage
----------------

All of the integrals used in the code, is stored in slowquant/temp
