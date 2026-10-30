#!/usr/bin/env bash
set -euo pipefail

VENV_BIN="${QE_VENV_BIN:-/home/hcf/project/QE/.venv/bin}"
DATA_ROOT="${QE_WIDERFACE_ROOT:-/home/hcf/project/QE/data/widerface_head_v1}"
RAW_ROOT="${DATA_ROOT}/raw/widerface"

mkdir -p "${RAW_ROOT}"

download_mirror() {
  local filename="$2"
  local expected_md5="$3"
  local expected_bytes="$4"
  local target="${RAW_ROOT}/${filename}"

  if [[ -f "${target}" ]] && echo "${expected_md5}  ${target}" | md5sum --check --status; then
    echo "[download] reuse ${target}"
  else
    echo "[download] ${filename}"
    while [[ ! -f "${target}" ]] || (( $(stat -c %s "${target}") < expected_bytes )); do
      curl --fail --location --continue-at - --max-time 120 \
        "https://data.brainchip.com/dataset-mirror/widerface/${filename}" \
        --output "${target}" || true
      echo "[download] ${filename} $(stat -c %s "${target}" 2>/dev/null || echo 0)/${expected_bytes} bytes"
    done
    echo "${expected_md5}  ${target}" | md5sum --check
  fi
}

download_http() {
  local url="$1"
  local filename="$2"
  local expected_md5="$3"
  local target="${RAW_ROOT}/${filename}"

  if [[ -f "${target}" ]] && echo "${expected_md5}  ${target}" | md5sum --check --status; then
    echo "[download] reuse ${target}"
  else
    echo "[download] ${filename}"
    curl --fail --location --retry 5 --continue-at - "${url}" --output "${target}"
    echo "${expected_md5}  ${target}" | md5sum --check
  fi
}

download_mirror "unused" "WIDER_train.zip" "3fedf70df600953d25982bcd13d91ba2" "1465602149"
download_mirror "unused" "WIDER_val.zip" "dfa7d7e790efa35df3788964cf0bbaea" "362752168"
download_mirror "unused" "wider_face_split.zip" "0e3767bcf0e326556d407bf5bff5d27c" "3591642"

for archive in WIDER_train.zip WIDER_val.zip wider_face_split.zip; do
  marker="${RAW_ROOT}/.${archive}.extracted"
  if [[ ! -f "${marker}" ]]; then
    echo "[extract] ${archive}"
    unzip -q "${RAW_ROOT}/${archive}" -d "${RAW_ROOT}"
    touch "${marker}"
  fi
done

echo "[complete] ${RAW_ROOT}"
