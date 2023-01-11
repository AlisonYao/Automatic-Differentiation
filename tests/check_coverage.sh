#!/usr/bin/env bash
# File       : check_coverage.sh
# Description: Coverage wrapper around test suite driver script
# Copyright 2022 Harvard University. All Rights Reserved.
set -e

pytest --cov=../best_autodiff --cov-report=html
pytest --cov=../best_autodiff --cov-report=term-missing | grep "^TOTAL" | grep -o '[^ ]*%' |sed 's/%//' |awk '{ sum += $1 } END { print sum/NR }' > cov.txt
coverage report -m

n=$(awk '{ sum += $1 } END { print sum/NR }' cov.txt)
rm cov.txt

thresh=90
if [ "$n" -lt "$thresh" ]; then
  exit 1
else
  exit 0
fi