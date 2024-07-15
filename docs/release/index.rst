:orphan:

***********
API changes
***********

We don't use semantic versioning.  The first number indicates that we have
made a major API break (e.g., 1.x to 2.x), which has happened once and probably
won't happen again for some time.  The point releases are new versions and may
contain minor API breakage.  Usually, this happens after a one cycle deprecation
period.

.. warning::
   Since we don't normally make bug-fix only releases, it may not make sense
   for you to use ``~=`` as a pip version specifier.

.. toctree::
   :maxdepth: 2

   api_0.1
