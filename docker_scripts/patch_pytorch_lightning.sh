#!/bin/bash

set -xueo pipefail

if [[ -f /tmp/ds/pytorch_lightning_torch2_compat.patch ]]
then
    patch -p0 -d/ </tmp/ds/pytorch_lightning_torch2_compat.patch
else
    patch -p0 -d/ </pytorch_lightning_torch2_compat.patch
fi