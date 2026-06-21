#!/bin/bash
# 将集群 VASP PBE 赝势库软链接成 pymatgen 期望的命名布局。
# pymatgen: <PMG_VASP_PSP_DIR>/POT_GGA_PAW_PBE/<symbol>/POTCAR
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="/share/apps/vasp/potentials/potpaw_PBE"
MIRROR="$HERE/psp_mirror"
mkdir -p "$MIRROR"
ln -sfn "$SRC" "$MIRROR/POT_GGA_PAW_PBE"
echo "POTCAR mirror ready: $MIRROR/POT_GGA_PAW_PBE -> $SRC"
ls "$MIRROR/POT_GGA_PAW_PBE" | head -3
