# Third-party notices

The physics code in `src/eamxx_torch/` is translated from E3SM. The pinned
source commit is recorded in `docs/plan.md`. Each translated file names its
source file(s).

## E3SM (EAMxx physics: SHOC, P3, cloud fraction, SPA, RRTMGP interface, IOP forcing, ZM, TMS; and share code, including `share/util/shr_orb_mod.F90` and `share/util/shr_flux_mod.F90`)

Source: https://github.com/E3SM-Project/E3SM, `components/eamxx/`.
E3SM is released under the following 3-Clause BSD license
(verbatim from E3SM's top-level `LICENSE`):

```
Copyright 2024, UChicago Argonne, LLC All Rights Reserved

Software Name: Energy Exascale Earth System Model (E3SM)

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this 
list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, 
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
3. Neither the names of UChicago Argonne, LLC or the Department of Energy nor 
the names of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

THE SOFTWARE IS SUPPLIED "AS IS" WITHOUT WARRANTY OF ANY KIND.
 
NEITHER THE UNITED STATES GOVERNMENT, NOR THE UNITED STATES DEPARTMENT OF
ENERGY, NOR UCHICAGO ARGONNE, LLC, NOR ANY OF THEIR EMPLOYEES, MAKES ANY
WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY
FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF ANY INFORMATION, DATA,
APPARATUS, PRODUCT, OR PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE WOULD
NOT INFRINGE PRIVATELY OWNED RIGHTS.
```

E3SM notes it was developed from CESM1.3beta10; CESM is public domain software.
The ZM scheme originates in CAM (CESM).

## RTE+RRTMGP (radiation)

Source: E3SM fork https://github.com/E3SM-Project/rte-rrtmgp, used by EAMxx.
Note: E3SM's top-level `LICENSE` table lists rrtmgp as "non-commercial". The
license file shipped in the rrtmgp source itself is BSD-3, reproduced here:

```
BSD 3-Clause License

Copyright (c) 2015-2018, Atmospheric and Environmental Research and Regents of the University of Colorado.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## YAKL (random-number generator used by RRTMGP's McICA sampling)

Source: https://github.com/mrnorman/YAKL, `src/YAKL_random.h`, as copied into
rte-rrtmgp's `cpp/rrtmgp_conversion.h` (`conv::Random`). Translated in
`src/eamxx_torch/rrtmgp/mcica.py`.

```
BSD 2-Clause License

Copyright (c) 2019, Matt Norman
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## EKAT (math utilities used by EAMxx)

Source: https://github.com/E3SM-Project/EKAT. Boost Software License 1.0
(Kokkos parts: see EKAT's LICENSE.txt). Only small numerical helpers, if any,
are translated.
