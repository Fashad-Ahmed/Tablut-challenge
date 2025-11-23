#! /bin/bash

if [[ $# -ne 3 ]] ; then
    echo "Need all arguments"
    echo "white or black as first parameter"
    echo "timeout as second parameter"
    echo "server ip as third parameter"
    echo "Example: $0 white 60 192.168.20.254"
    exit 1
fi

if [[ $1 != "white" && $1 != "black" ]] ; then
    echo "First parameter must be white or black"
    exit 1
fi

if [[ $2 -le 0 ]] ; then
    echo "Timeout must be greater than 0"
    exit 1
fi

source venv/bin/activate
python3 -m python_client.agent "$1" "$2" "$3" --model models/rl_value_net_5M.zip --depth 4