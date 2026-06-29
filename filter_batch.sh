#!/usr/bin/env bash

set -euo pipefail

thisFile="$(realpath "${BASH_SOURCE[0]}")"
thisDir="$(dirname "${thisFile}")"

function main() {
    csvFile="${1:-}"

    if [ ! -f "${csvFile}" ]; then
        echo "Usage: ${thisFile} <csv_file>"
        return 1
    fi

    dataDir="${thisDir}/data"
    treesDir="${dataDir}/trees"

    files="$(cat "${csvFile}" | cut -d';' -f1 | tail -n +2)"

    filteredDir="${dataDir}/filtered"
    mkdir -p "${filteredDir}"

    while read -r file; do
        inPath="${treesDir}/${file}"
        outPath="${filteredDir}/${file}"

        if [ ! -f "${inPath}" ]; then
            echo "File ${inPath} does not exist"
            return 1
        fi

        cp "${inPath}" "${outPath}"
    done <<< "${files}"
}

if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    main "$@"
fi
