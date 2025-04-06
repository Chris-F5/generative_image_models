#!/bin/bash

DIR=$(realpath "$0" | xargs dirname)

set -e

aws ec2 request-spot-instances \
    --instance-count 1 \
    --region eu-central-1 \
    --launch-specification "file://$DIR/aws-spot-config.json"
