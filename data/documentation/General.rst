
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
   
Inside the integral call, the basisset file is reconstructed into three different arrays, containing the basisset information. The first one is basisidx that have the following form:

.. math::
   \left[\begin{array}{cc}\mathrm{\#primitives} & \mathrm{loop\,start\,idx}\end{array}\right]
   
It thus contains the number of primitives in each basisfunction, and what start index it have for loop inside the integral code.

The second array is basisint, that have the following forms:

.. math::
   \left[\begin{array}{ccc}l & m & n\end{array}\right]

.. math::
   \left[\begin{array}{cccc}l & m & n & \mathrm{atom\,idx}\end{array}\right]

The first one is for regular integrals and the second one is for derivatives. Both contains all the angular momentum quantum numbers, and the derivative also contains the atom index (used in derivative of VNe).

The last array is basisfloat and have the following forms:

.. math::
   \left[\begin{array}{cccccc}N & \zeta & c & x & y & z\end{array}\right]

.. math::
   \left[\begin{array}{cccccccccccc}N & \zeta & c & x & y & z & N_{x,+} & N_{x,-} & N_{y,+} & N_{y,-} & N_{z,+} & N_{z,-}\end{array}\right]
   
basisfloat contains the normalization constants, Gaussian exponent and prefacor and the coordinates of the atoms. The second one is again for the derivatives, it contains normalization constants of the differentiated primitives.

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

The results are parsed around in the code as a dictionary. It contains the results from different calculatations that are either used in other calculations or given as an output.
